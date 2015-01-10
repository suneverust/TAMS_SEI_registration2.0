/*********************************************
 * Author: Bo Sun                            *
 * Afflication: TAMS, University of Hamburg  *
 * E-Mail: bosun@informatik.uni-hamburg.de   *
 *         user_mail@QQ.com                  *
 * Date: Nov 13, 2014                        *
 * Licensing: GNU GPL license.               *
 *********************************************/

#ifndef TAMS_SEI_REGISTRATION_H_
#define TAMS_SEI_REGISTRATION_H_

#ifndef TYPE_DEFINITION_
#define TYPE_DEFINITION_
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;
#endif /*TYPE_DEFINITION_*/

/** \brief cart2sph(X,Y,Z, azimuth, polar) transforms Cartesian coordinates stored in
  * corresponding elements of arrays X, Y, and Z into spherical coordinates.
  * azimuth and polar are angular displacements in radians.
  * azimuth(longitudinal) is the counterclockwise angle in the x-y plane
  * measured from the positive x-axis.
  * polar(colatitudianl) is the polar angle measured from the z axis.
  *
  * 0 < azimuth < 2*M_PI; 0 < polar < M_PI
  */
void tams_cart2sph(float x, float y, float z,
					float& azimuth, float& polar);

/** \brief tams_vector_normalization normalize the input vector
  * Parameters:
  * [in]     tams_vector   the input vector
  * [out]    tams_vector   the normalized vector (values are in range [0,1])
  */
void tams_vector_normalization (std::vector<float> &tams_vector);

/** \brief tams_vector2entropy compute the entropy of a vector
  * Parameters:
  * [in]   tams_vector     the input vector
  * [in]   hist_bin        the size of histogram in entropy computation
  * [out]  entropy         the resultant entropy
  */
void tams_vector2entropy( const std::vector<float> tams_vector,
                          const size_t hist_bin,
                          float& entropy);

/** \brief computeSEI calculate the SEI of the input point cloud
  * Parameters:
  * [in]     cloud         the input vector
  * [in]     sei_dim       the dimension of SEI(sei_dim X sei_dim)
  * [in]     hist_bin      the size of histogram in entropy computation
  * [out]    entropy       the resulstant SEI stored in a (sei_dim X sei_dim) matrix
  */
void computeSEI( const PointCloudT cloud,
                 size_t sei_dim,
                 size_t hist_bin,
                 Eigen::MatrixXf &entropy);

/** \brief PhaseCorrelation3D compute the offset between two input volumes
  * based on POMF (Phase Only Matched Filter)
  * -->> Q(k) = conjugate(S(k))/|S(k)| * R(k)/|R(k)|
  * -->> q(x) = ifft(Q(k))
  * -->> (xs,ys) = argmax(q(x))
  * Note that the storage order of FFTW is row-order
  * We adopt the RIGHT-hand Cartesian coordinate system.
  * Parameters:
  * [in] signal              the input(signal) volume
  * [in] pattern             the input(pattern) volume
  * [in] height              the height of input volumes(range of x)
  * [in] width               the width of input volumes (range of y)
  * [in] depth               the depth of input volumes (range of z)
  * [out] height_offset      the result offset, we move down (positive x axis) pattern
  *                          height_offset to match signal
  * [out] width_offset       the result offset, we move right (positive y axis) pattern
  *                          width_offset to match signal
  * [out] depth_offset       the result offset, we move close to viewer (positive z axis) pattern
  *                          depth_offset to match signal
  */
void PhaseCorrelation3D(const double *signal,
                        const double *pattern,
                        const int height,
                        const int width,
                        const int depth,
                        int &height_offset,
                        int &width_offset,
                        int &depth_offset);

/** \brief voxelsize2volumesize compute the size of volume
  * the points cloud rendered based on the size of voxel.
  * Parameters:
  * [in]  cloud         the input point cloud
  * [in]  voxelsize     the size of the voxel
  * [out] volumesize    the size of volume the input cloud should be rendered to
  */
void voxelsize2volumesize ( const PointCloudT cloud,
                            Eigen::Vector3f voxelsize,
                            Eigen::Vector3i &volumesize);

/** \brief point2volume render the input point cloud
  * to a volume, assign the number of points (or max curvature of points)
  * in a voxel to the value of voxel.
  * Parameters:
  * [in]  cloud              the input point cloud
  * [in]  voxelsize          the size of the voxel
  * [in]  volumesize         the real size of resutant volume
  * [in]  volumesize_origin  the volumesize calculated by function "voxelsize2volumesize"
  * [out] volume             the resultant volume contained in the
  *                          volumesize(0)*volumesize(1)*volumesize(2) matrix
  */
void point2volume (const PointCloudT cloud,
                   Eigen::Vector3f voxelsize,
                   Eigen::Vector3i volumesize,
                   Eigen::Vector3i volumesize_origin,
                   double *volume);
#endif /*TAMS_SEI_REGISTRATION_H_*/
