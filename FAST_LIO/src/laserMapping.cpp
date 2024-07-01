// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// #include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver2/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include <pcl/kdtree/kdtree_flann.h>
//添加的头文件
#include <map>
#include <unordered_map>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
//添加关键帧和id话题
string map_file_path, lid_topic, imu_topic, keyFrame_topic, keyFrame_id_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool    recontructKdTree = false;
bool    updateState = false;
int     updateFrequency = 100;



vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;//状态，噪声维度，输入
state_ikfom state_point;                        //状态
vect3 pos_lid;

nav_msgs::Path path, path_updated;/*发布更新的状态路径*/
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose, msg_body_pose_updated;

//定义指向激光雷达预处理类preprocess的指针
shared_ptr<Preprocess> p_pre(new Preprocess());
//指向IMU数据预处理类imu_process的指针
shared_ptr<ImuProcess> p_imu(new ImuProcess());

/*添加关键帧*/
vector<PointCloudXYZI::Ptr> cloudKeyFrames;  // 存放历史的关键帧点云
queue< pair<uint32_t, PointCloudXYZI::Ptr> > cloudBuff;// 缓存部分历史的lidar帧，用于提取出关键帧点云
vector<uint32_t> idKeyFrames;           // keyframes 的 id
queue<uint32_t> idKeyFramesBuff;         // keyframes 的 id buffer
nav_msgs::Path pathKeyFrames;           // keyframes
uint32_t data_seq;                        // 数据的序号
uint32_t lastKeyFramesId;               // 最新关键帧对应里程计的ID
geometry_msgs::Pose lastKeyFramesPose;  // 最新关键帧的位姿（世界到imu）
vector<geometry_msgs::Pose> odoms;

/*维护submap*/
pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtreeSurroundingKeyPoses(new pcl::KdTreeFLANN<pcl::PointXYZ>()); // 周围关键帧pose的kdtree
pcl::VoxelGrid<pcl::PointXYZ> downSizeFilterSurroundingKeyPoses;

//唤醒所有线程
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp) 
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

/*两点之间距离，用于判断关键帧*/
float pointDistance(pcl::PointXYZ p1, pcl::PointXYZ p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}


//body系转到world系，通过ikfom的位置和姿态
void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    //下面式子最里面括号是从雷达到imu坐标系，然后从imu转到世界坐标系
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

//把点从body系转到world系
void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

//含有rgb的点云从body系转到world系
void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

//得到被剔除的点（重构IKD-Tree过滤后剔除的点集）
/*修改*/
void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);//返回被剔除的点
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();//清空需要移除的区域
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);//好像没用到
    //global系下的lidar位置
    V3D pos_LiD = pos_lid;
    //若未初始化，以当前w系下雷达点为中心设置200*200*200的局部地图
    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    //lidar与立方体盒子六个面的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    //分别在xyz三个维度上到地图边缘的距离
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        //与某个方向上的边界距离太小，标记需要移除need_move，
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    //不需要移动就退出，以下是地图移动的操作
    if (!need_move) return;

    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    //新的局部地图盒子的边界点
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);//移除较远包围盒
        } 
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    //使用Boxs删除指定盒内的点
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

//除avia类型之外的雷达点云回调
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);   //点云预处理
    lidar_buffer.push_back(ptr);//预处理后的点云放入buffer
    time_buffer.push_back(msg->header.stamp.toSec());//时间戳放入buffer
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();//唤醒所有时间
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;//时间同步flag，false表示未进行时间同步，true相反
//接收livox点云的回调
void livox_pcl_cbk(const livox_ros_driver2::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    //如果当前扫描的激光雷达数据的时间戳比上一次扫描的时间戳早，需要将队列清空
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    //如果不需要进行时间同步，而imu时间戳和雷达时间戳相差大于10s，则输出错误信息
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    //如果需要进行时间同步，而imu时间戳和雷达时间戳相差大于1s，则进行时间同步
    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    //用pcl点云格式保存接收到的激光雷达数据
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    //对激光雷达数据进行预处理（特征提取或者降采样），p_pre是preprocess类的智能指针
    p_pre->process(msg, ptr);//预处理，包括特征提取，降采样等操作
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


//imu回调
/*修改407 */
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    // msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    //将imu和激光雷达时间对齐
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    //当前时间戳小于上一时刻时间戳，则清空队列
    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void keyFrame_cbk(const nav_msgs::Path::ConstPtr &msg_keyframes){
    // 更新关键帧
    pathKeyFrames = *msg_keyframes;
}

void keyFrame_id_cbk(const std_msgs::Header::ConstPtr &msg_keyframe_id){
    // 将订阅到的关键帧id先加到idKeyFramesBuff中
    idKeyFramesBuff.push(msg_keyframe_id->seq);
}


double lidar_mean_scantime = 0.0;
int    scan_num = 0;
//将两帧雷达点云时间内的imu数据从队列中取出，时间对齐后放入meas
/*修改：459 462*/
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        //从点云队列取数据放入meas中
        meas.lidar = lidar_buffer.front();
        // meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_buffer.pop_front();
            return false;
            // lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            // ROS_WARN("Too few input point cloud!\n");
        }
        // else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        // {
        //     lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        // }
        // else
        // {
        //     scan_num ++;
        //     lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
        //     lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        // }

        // meas.lidar_end_time = lidar_end_time;
        meas.lidar_beg_time = time_buffer.front();
        lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    // 拿出lidar_beg_time到lidar_end_time之间的所有IMU数据
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
/*修改：*/
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        //dense_pub_en为真无需降采样
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        // static int scan_wait_num = 0;
        // scan_wait_num ++;
        // if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        // {
        //     pcd_index ++;
        //     string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
        //     pcl::PCDWriter pcd_writer;
        //     cout << "current scan saved to /PCD/" << all_points_dir << endl;
        //     pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
        //     pcl_wait_save->clear();
        //     scan_wait_num = 0;
        // }
    }
}

//把去畸变后的点云转到imu系
void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    laserCloudmsg.header.seq = data_seq;
    pubLaserCloudFull_body.publish(laserCloudmsg);
    cloudBuff.push( pair<int, PointCloudXYZI::Ptr>(data_seq ,laserCloudIMUBody) );  // 缓存所有发给后端的点云
    publish_count -= PUBFRAME_PERIOD;
}


/*将未降采样的点云作为lidar系发布出去，此为scancontext使用*/
void publish_frame_lidar(const ros::Publisher & pubLaserCloudFull_lidar)
{
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*feats_undistort, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "lidar";
    pubLaserCloudFull_lidar.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

//把起作用的特征点转到地图中
void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));//创建一个点云用于储存转换到世界坐标系的点云，从h_share_model中获得
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

//发布ikd-tree地图
void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

//设置输出的t，q，在publish_odometry,publish_path中调用
template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);//将eskf求得的位置传入
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;//将eskf求得的姿态传入
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}


void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.header.seq = data_seq;
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);

    odoms.push_back(odomAftMapped.pose.pose);

    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        //设置协方差P里面先是旋转后是位置，这个POSE里面先是位置后是旋转 
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );//；发布tf变换
}


void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    // if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}


//计算残差信息
/*修改866 874*/
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); //将body系的有效点云存储清空
    corr_normvect->clear(); //将对应的法向量清空
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    /*最接近曲面搜索和残差计算*/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    //对降采样后的每个特征点进行残差计算
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i];//获取降采样后的每个特征点 
        PointType &point_world = feats_down_world->points[i];//获取降采样后的每个特征点的世界坐标 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        //将点转到世界坐标系下再进行残差计算
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        //如果收敛了
        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            //在已构造的地图上查找特征点的最近邻
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            //如果最近邻的点数小于NUM_MATCH_POINTS或者最近邻点到特征点距离大于5，则认为该点不是有效点
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;//平面点信息
        point_selected_surf[i] = false;//将该点设置为无效点，用来计算是否为平面点
        //拟合平面方程ax+by+cz+d=0并求解点到平面距离
        if (esti_plane(pabcd, points_near, 0.1f))//esti_plane函数找平面法向量
        {
            //求点到面的距离
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());//计算残差

            if (s > 0.9)//如果残差大于阈值，则认为该点是有效点
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;//将点到面的距离储存至normvec的intensity中
                res_last[i] = abs(pd2);//将残差储存至res_last中
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        //根据point_selected_surf状态判断哪些点可用
        if (point_selected_surf[i])
        {
            //将降采样后的每个特征点储存至laserCloudOri
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            //拟合平面点存到corr_normvect中
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];//计算总残差
            effct_feat_num ++;//有效特征点+1
        }
    }

    // if (effct_feat_num < 1)
    // {
    //     ekfom_data.valid = false;
    //     ROS_WARN("No Effective Points! \n");
    //     return;
    // }

    res_mean_last = total_residual / effct_feat_num;//计算平均残差
    match_time  += omp_get_wtime() - match_start;   
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23//测量雅克比矩阵H
    ekfom_data.h.resize(effct_feat_num);
    
    //求观测值与误差的雅克比矩阵，如论文式14以及式12、13
    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        //将点值转换到叉乘矩阵（叉乘矩阵是一种用于表示三维空间中的旋转和转换的矩阵）
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        //转换到imu坐标系下
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        //imu系下点的叉乘矩阵
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        //得到对应的曲面/角的法向量
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        //计算测量雅克比矩阵H
        V3D C(s.rot.conjugate() *norm_vec);//旋转矩阵的转置与法向量相乘得到C
        V3D A(point_crossmat * C);//imu系下的叉乘矩阵乘C得到A
        // if (extrinsic_est_en)
        // {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        // }
        // else
        // {
        //     ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        // }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    data_seq = 0;
    //添加参数
    nh.param<int>("updateFrequency", updateFrequency, 100);
    nh.param<bool>("updateState", updateState, true);
    nh.param<bool>("recontructKdtree", recontructKdTree, true);
    nh.param<string>("common/keyFrame_topic", keyFrame_topic, "/aft_pgo_path");
    nh.param<string>("common/keyFrame_id_topic", keyFrame_id_topic, "/key_frames_ids");



    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<string>("map_file_path", map_file_path, "");
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    nh.param<double>("mapping/fov_degree", fov_deg, 180);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<double>("preprocess/blind",  p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    //添加的path_update
    path_updated.header.stamp    = ros::Time::now();
    path_updated.header.frame_id ="camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    // FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    // HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    // memset(point_selected_surf, true, sizeof(point_selected_surf));
    // memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);

    ros::Subscriber sub_keyframes = nh.subscribe(keyFrame_topic, 10, keyFrame_cbk);
    ros::Subscriber sub_keyframes_id = nh.subscribe(keyFrame_id_topic, 10, keyFrame_id_cbk);

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);

    /*发布更新的状态路径*/
    ros::Publisher pubPath_updated  = nh.advertise<nav_msgs::Path>
            ("/path_updated", 100000);
    ros::Publisher pubLaserCloudFull_lidar = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_lidar", 100000);
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    uint32_t count = 1;
    while (status)
    {
        //如果有中断产生，则结束循环
        if (flg_exit) break;
        ros::spinOnce();

        // 接收关键帧, 一直循环直到其中一个为空（理论上应该是idKeyFramesBuff先空）
        {
            while( !cloudBuff.empty() && !idKeyFramesBuff.empty() ){
                while( idKeyFramesBuff.front() > cloudBuff.front().first )
                {
                    cloudBuff.pop();
                }
                // 此时idKeyFramesBuff.front() == cloudBuff.front().first
                assert(idKeyFramesBuff.front() == cloudBuff.front().first);
                idKeyFrames.push_back(idKeyFramesBuff.front());
                cloudKeyFrames.push_back( cloudBuff.front().second );
                idKeyFramesBuff.pop();
                cloudBuff.pop();
            }
            assert(pathKeyFrames.poses.size() <= cloudKeyFrames.size() );   // 有可能id发过来了，但是节点还未更新
            
            // 记录最新关键帧的信息
            if(pathKeyFrames.poses.size() >= 1){
                lastKeyFramesId = idKeyFrames[pathKeyFrames.poses.size() - 1];
                lastKeyFramesPose = pathKeyFrames.poses.back().pose;
            }
        }

        

        //将激光雷达点云数据和imu数据从缓存队列中取出，进行时间对齐，并保存到measures中
        if(sync_packages(Measures)) 
        {
            //激光第一次扫描
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            //imu数据预处理，其中包含了点云去畸变处理，前向传播，反向传播
            p_imu->Process(Measures, kf, feats_undistort);
            //获取kf预测的全局状态（imu）
            state_point = kf.get_x();
            //世界坐标系下雷达位置
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                // first_lidar_time = Measures.lidar_beg_time;
                // p_imu->first_lidar_time = first_lidar_time;
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            //判断是否初始化完成，条件为第一个点云时间与第一帧扫描起始时间的差值大于INIT_TIME
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;

            
            if(count % updateFrequency == 0 )
            {
                
                count = 1;
                if(recontructKdTree && pathKeyFrames.poses.size() > 20)
                {
                    
                    // /*** 所有关键帧的地图 ***/
                    // PointCloudXYZI::Ptr keyFramesMap(new PointCloudXYZI());
                    // PointCloudXYZI::Ptr keyframesTmp(new PointCloudXYZI());
                    // Eigen::Isometry3d poseTmp;
                    // assert(pathKeyFrames.poses.size() <= cloudKeyFrames.size() );   // 有可能id发过来了，但是节点还未更新
                    // int keyFramesNum = pathKeyFrames.poses.size();
                    // for(int i = 0; i < keyFramesNum; ++i){
                    //     downSizeFilterMap.setInputCloud(cloudKeyFrames[i]);
                    //     downSizeFilterMap.filter(*keyframesTmp);
                    //     tf::poseMsgToEigen(pathKeyFrames.poses[i].pose,poseTmp);
                    //     pcl::transformPointCloud(*keyframesTmp , *keyframesTmp, poseTmp.matrix());
                    //     *keyFramesMap += *keyframesTmp;
                    // }
                    // downSizeFilterMap.setInputCloud(keyFramesMap);
                    // downSizeFilterMap.filter(*keyFramesMap);

                    // ikdtree.reconstruct(keyFramesMap->points);

                    /*** 距离近的关键帧构成的子图 ***/
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudKeyPoses3D(new pcl::PointCloud<pcl::PointXYZ>());    // 历史关键帧位姿（位置）
                    pcl::PointCloud<pcl::PointXYZ>::Ptr surroundingKeyPoses(new pcl::PointCloud<pcl::PointXYZ>());    
                    pcl::PointCloud<pcl::PointXYZ>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<pcl::PointXYZ>());    
                    
                    //将最新关键帧加入cloudKeyPoses3D
                    for(auto keyFramePose:pathKeyFrames.poses)
                    {
                        cloudKeyPoses3D->points.emplace_back(keyFramePose.pose.position.x, 
                                                                keyFramePose.pose.position.y, 
                                                                keyFramePose.pose.position.z);
                    }
                    double surroundingKeyframeSearchRadius = 5;
                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqDis;
                    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); 
                    kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
                    // 遍历搜索结果，pointSearchInd存的是结果在cloudKeyPoses3D下面的索引
                    unordered_map<float, int> keyFramePoseMap;  // 以pose的x坐标为哈希表的key
                    for (int i = 0; i < (int)pointSearchInd.size(); ++i)
                    {
                        int id = pointSearchInd[i];
                        // 加入相邻关键帧位姿集合中
                        surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
                        keyFramePoseMap[cloudKeyPoses3D->points[id].x] = id;
                    }

                    // 降采样一下
                    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
                    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

                    // 加入与当前关键帧靠近的offset个帧，这些帧加进来是合理的
                    int numPoses = cloudKeyPoses3D->size();
                    int offset = 10;
                    for (int i = numPoses-1; i >= numPoses-1 - offset && i >= 0; --i)
                    {
                        surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
                        keyFramePoseMap[cloudKeyPoses3D->points[i].x] = i;
                    }

                    // 将相邻关键帧集合对应的点加入到局部map中，作为scan-to-map匹配的局部点云地图
                    // PointCloudXYZI::Ptr keyFramesSubmap = extractCloud(surroundingKeyPosesDS, keyFramePoseMap);
                    
                    PointCloudXYZI::Ptr keyFramesSubmap(new PointCloudXYZI());
                    // 遍历当前帧（实际是取最近的一个关键帧来找它相邻的关键帧集合）时空维度上相邻的关键帧集合
                    for (int i = 0; i < (int)surroundingKeyPosesDS->size(); ++i)
                    {
                        ROS_INFO("surroundingKeyPosesDS->points[i].x: %f", surroundingKeyPosesDS->points[i].x);
                        ROS_INFO("surroundingKeyPosesDS->points[i].x: %d", keyFramePoseMap.size());
                        // assert(keyFramePoseMap.count(surroundingKeyPosesDS->points[i].x) != 0);
                        if(keyFramePoseMap.count(surroundingKeyPosesDS->points[i].x) == 0)
                            continue;
                        
                        // 距离超过阈值，丢弃
                        if (pointDistance(surroundingKeyPosesDS->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)    // 丢弃那些满足时间临近，不满足空间临近的点
                            continue;

                        // 相邻关键帧索引
                        int thisKeyInd = keyFramePoseMap[ surroundingKeyPosesDS->points[i].x ];  // 以intensity作为红黑树的索引
                        
                        PointCloudXYZI::Ptr keyframesTmp(new PointCloudXYZI());
                        Eigen::Isometry3d poseTmp;
                        assert(pathKeyFrames.poses.size() <= cloudKeyFrames.size() );   // 有可能id发过来了，但是节点还未更新
                        int keyFramesNum = pathKeyFrames.poses.size();
                        
                        downSizeFilterMap.setInputCloud(cloudKeyFrames[thisKeyInd]);
                        downSizeFilterMap.filter(*keyframesTmp);

                        tf::poseMsgToEigen(pathKeyFrames.poses[thisKeyInd].pose,poseTmp);
                        pcl::transformPointCloud(*keyframesTmp , *keyframesTmp, poseTmp.matrix());
                        *keyFramesSubmap += *keyframesTmp;
                    }
                    downSizeFilterMap.setInputCloud(keyFramesSubmap);
                    downSizeFilterMap.filter(*keyFramesSubmap);

                    //重构ikd树
                    ikdtree.reconstruct(keyFramesSubmap->points);
                }

                // 更新状态
                if(updateState)
                {
                    
                    state_ikfom state_updated = kf.get_x();
                    Eigen::Isometry3d lastPose(state_updated.rot);
                    lastPose.pretranslate(state_updated.pos);
                    

                    Eigen::Isometry3d lastKeyFramesPoseEigen;       // 最新的关键帧位姿
                    tf::poseMsgToEigen(lastKeyFramesPose, lastKeyFramesPoseEigen);
                    

                    Eigen::Isometry3d lastKeyFrameOdomPoseEigen;    // 最新的关键帧对应的odom的位姿
                    tf::poseMsgToEigen(odoms[lastKeyFramesId], lastKeyFrameOdomPoseEigen);
                    
                    // lastPose表示世界坐标系到当前坐标系的变换，下面两个公式等价
                    // lastPose = (lastKeyFramesPoseEigen.inverse() * lastKeyFrameOdomPoseEigen* lastPose.inverse()).inverse();
                    lastPose = lastPose * lastKeyFrameOdomPoseEigen.inverse() * lastKeyFramesPoseEigen;

                    geometry_msgs::Pose msgtmp;
                    tf::poseEigenToMsg(lastKeyFramesPoseEigen,msgtmp);
                    Eigen::Quaterniond lastPoseQuat( lastPose.rotation() );
                    Eigen::Vector3d lastPoseQuatPos( lastPose.translation() );
                    state_updated.rot = lastPoseQuat;
                    state_updated.pos = lastPoseQuatPos;
                    kf.change_x(state_updated);
                    
                    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov P_updated = kf.get_P();  // 获取当前的状态估计的协方差矩阵
                    P_updated.setIdentity();
                    // QUESTION: 状态的协方差矩阵是否要更新为一个比较的小的值？ 
                    // init_P(0,0) = init_P(1,1) = init_P(2,2) = 0.00001; 
                    // init_P(3,3) = init_P(4,4) = init_P(5,5) = 0.00001;
                    P_updated(6,6) = P_updated(7,7) = P_updated(8,8) = 0.00001;
                    P_updated(9,9) = P_updated(10,10) = P_updated(11,11) = 0.00001;
                    P_updated(15,15) = P_updated(16,16) = P_updated(17,17) = 0.0001;
                    P_updated(18,18) = P_updated(19,19) = P_updated(20,20) = 0.001;
                    P_updated(21,21) = P_updated(22,22) = 0.00001; 
                    kf.change_P(P_updated);

                    msg_body_pose_updated.pose.position.x = state_updated.pos(0);
                    msg_body_pose_updated.pose.position.y = state_updated.pos(1);
                    msg_body_pose_updated.pose.position.z = state_updated.pos(2);
                    msg_body_pose_updated.pose.orientation.x = state_updated.rot.x();
                    msg_body_pose_updated.pose.orientation.y = state_updated.rot.y();
                    msg_body_pose_updated.pose.orientation.z = state_updated.rot.z();
                    msg_body_pose_updated.pose.orientation.w = state_updated.rot.w();
                    msg_body_pose_updated.header.stamp = ros::Time().fromSec(lidar_end_time);
                    msg_body_pose_updated.header.frame_id = "camera_init";

                    /*** if path is too large, the rvis will crash ***/
                    static int jjj = 0;
                    jjj++;
                    // if (jjj % 10 == 0)
                    {
                        path_updated.poses.push_back(msg_body_pose_updated);
                        pubPath_updated.publish(path_updated);
                    }
                }
            }
            ++count;

            /*** Segment the map in lidar FOV ***/
            //动态调整局部地图，在拿到eskf前馈后
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            //去畸变，降采样
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            //构建ikd树
            if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    //设置ikd树的降采样参数
                    ikdtree.set_downsample_param(filter_size_map_min);
                    //将降采样得到的地图点大小与body系大小一致
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));//将下采样得到的地图点转换为世界坐标系下的点云
                    }
                    //组织ikd树
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            //获取ikd树中的有效节点数，无效点打上deleted标签
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            /*拿当前帧与ikd-tree建立的地图算出的残差，然后计算更新自己的位置，并将更新后的结果通过map-incremental传递给ikd-tree表示的映射中*/
            //外参，旋转矩阵转欧拉角
            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;//输出预测的结果写入文件

            if(0) // If you need to see map point, change to "if(1)"
            {
                //释放PCL_Storage内存
                PointVector ().swap(ikdtree.PCL_Storage);
                // 把树展平用于展示
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            //搜索索引
            pointSearchInd_surf.resize(feats_down_size);
            //将降采样处理后的点云用于搜索最近
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            //迭代卡尔曼滤波更新，更新地图信息
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            //发布里程计
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            //向映射ikdtree中添加特征点
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) 
            {
                publish_frame_body(pubLaserCloudFull_body);
                publish_frame_lidar(pubLaserCloudFull_lidar);

            }
            ++data_seq;
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        cout << "all points saved to " << all_points_dir<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
