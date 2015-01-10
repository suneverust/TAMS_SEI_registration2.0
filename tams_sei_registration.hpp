/*********************************************
 * Author: Bo Sun                            *
 * Afflication: TAMS, University of Hamburg  *
 * E-Mail: bosun@informatik.uni-hamburg.de   *
 *         user_mail@QQ.com                  *
 * Date: Nov 13, 2014                        *
 * Licensing: GNU GPL license.               *
 *********************************************/
#ifndef TAMS_SEI_REGISTRATION_HPP_
#define TAMS_SEI_REGISTRATION_HPP_

#include <cmath>
#include <algorithm>
#include <vector>
#include <eigen3/Eigen/Dense>

#include <pcl/io/io.h>
#include <pcl/common/common.h>

#include "fftw3.h"
#include "tams_sei_registration.h"

void tams_cart2sph(float x, float y, float z,
                   float& azimuth, float& polar)
{
    polar = atan2(hypot(x,y),z);
    azimuth = atan2(y,x);
    if (azimuth < 0)
        azimuth = azimuth + 2*M_PI;
}

void tams_vector_normalization (std::vector<float> &tams_vector)
{
    float max_element = (*std::max_element(tams_vector.begin(),tams_vector.end()));
    float min_element = (*std::min_element(tams_vector.begin(),tams_vector.end()));

    if (max_element == min_element)
        return;
    for (std::vector<float>::iterator itr = tams_vector.begin();
         itr != tams_vector.end(); itr ++)
    {
        // save memory but dangerous!
        (*itr)=((*itr)-min_element)/(max_element-min_element);
    }
}

void tams_vector2entropy (const std::vector<float> tams_vector,
                          const size_t hist_bin,
                          float &entropy)
{
    std::vector<float> temp_hist(hist_bin+1);
    for (std::vector<float>::const_iterator itr = tams_vector.begin();
         itr !=tams_vector.end(); itr++)
    {
        temp_hist[floor((*itr)*hist_bin)]++;
    }
    temp_hist[hist_bin-1]++;
    temp_hist.pop_back();
    if(temp_hist.size()!=hist_bin)
        pcl::console::print_warn(
                    "Warning: something maybe wrong in computing Histogram!\n");
    // Parzen Window: [0.05, 0.25, 0.40, 0.25, 0.05]
    std::vector<float> temp_hist_pad;
    temp_hist_pad.push_back(0.0);
    temp_hist_pad.push_back(0.0);
    for(std::vector<float>::iterator itr = temp_hist.begin();
        itr != temp_hist.end(); itr++)
    {
        temp_hist_pad.push_back(*itr);
    }
    temp_hist_pad.push_back(0.0);
    temp_hist_pad.push_back(0.0);
    std::vector<float>().swap(temp_hist);

    std::vector<float> tams_hist;
    for(std::vector<float>::iterator itr=temp_hist_pad.begin()+2;
        itr !=temp_hist_pad.end()-2; itr++)
    {
        tams_hist.push_back( (*(itr-2))*0.05
                            +(*(itr-1))*0.25
                            +(*(itr  ))*0.40
                            +(*(itr+1))*0.25
                            +(*(itr+2))*0.05);
    }
    if(tams_hist.size()!=hist_bin)
    {
        pcl::console::print_error("Error: Histogram Parzen Window Failed\n");
        return;
    }
    std::vector<float>().swap(temp_hist_pad);
    
	entropy = 0.0;
    for (std::vector<float>::iterator itr = tams_hist.begin();
         itr !=tams_hist.end(); itr++)
    {
        if ((*itr)>0)
            entropy += -(*itr)*log((*itr));
    }
    std::vector<float>().swap(tams_hist);
}

void computeSEI (const PointCloudT cloud,
                 size_t sei_dim,
                 size_t hist_bin,
                 Eigen::MatrixXf &entropy)
{
    float sei_azimuth_spa = 2*M_PI/(2*sei_dim);
    float sei_polar_spa   = M_PI/(2*sei_dim);

    // Voxel Division
    Eigen::Array<std::vector<float>, Eigen::Dynamic, Eigen::Dynamic>
            sei_points_divi(2*sei_dim, 2*sei_dim);

    float temp_az , temp_polar;
    size_t temp_sei_azth, temp_sei_polarth;
    double dist;
    for (PointCloudT::const_iterator itr=cloud.begin();
         itr!=cloud.end(); itr++)
    {
        dist = sqrt((*itr).x*(*itr).x+
                    (*itr).y*(*itr).y+
                    (*itr).z*(*itr).z);
        tams_cart2sph((*itr).x, (*itr).y, (*itr).z,
                      temp_az, temp_polar);
        if (temp_az < sei_azimuth_spa/2 ||
                temp_az >= 2*M_PI-sei_azimuth_spa/2)
            temp_sei_azth = 0;
        else
            temp_sei_azth = floor((temp_az-sei_azimuth_spa/2)/sei_azimuth_spa)+1;
        temp_sei_polarth = floor(temp_polar/sei_polar_spa);

        sei_points_divi(temp_sei_azth, temp_sei_polarth).push_back(dist);
    }

    // compute entropy
    for(temp_sei_azth = 0; temp_sei_azth < 2*sei_dim; temp_sei_azth++)
    {
        for(temp_sei_polarth = 0; temp_sei_polarth < 2*sei_dim; temp_sei_polarth++)
        {
            if (sei_points_divi(temp_sei_azth, temp_sei_polarth).size()<5)
            {
                entropy(temp_sei_azth, temp_sei_polarth) = 0;
                continue;
            }
            if (    (*std::max_element(sei_points_divi(temp_sei_azth, temp_sei_polarth).begin(),
                                       sei_points_divi(temp_sei_azth, temp_sei_polarth).end()))
                    ==
                    (*std::min_element(sei_points_divi(temp_sei_azth, temp_sei_polarth).begin(),
                                       sei_points_divi(temp_sei_azth, temp_sei_polarth).end())))
            {
                entropy(temp_sei_azth, temp_sei_polarth) = 0;
                continue;
            }

            tams_vector_normalization(sei_points_divi(temp_sei_azth, temp_sei_polarth));

            tams_vector2entropy(sei_points_divi(temp_sei_azth, temp_sei_polarth),
                                hist_bin,
                                entropy(temp_sei_azth, temp_sei_polarth));
        }
    }
}

void PhaseCorrelation3D(const double *signal,
                        const double *pattern,
                        const int height,
                        const int width,
                        const int depth,
                        int &height_offset,
                        int &width_offset,
                        int &depth_offset)
{
    int size = height*width*depth;
    fftw_complex *signal_volume  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*size);
    fftw_complex *pattern_volume = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*size);

    for (int i=0; i < size; i++)
    {
        signal_volume[i][0] = signal[i];
        signal_volume[i][1] = 0;
    }
    for (int j=0; j < size; j++)
    {
        pattern_volume[j][0] = pattern[j];
        pattern_volume[j][1] = 0;
    }

    // forward fft
    fftw_plan signal_forward_plan = fftw_plan_dft_3d (height, width, depth, signal_volume, signal_volume,
                                                    FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pattern_forward_plan  = fftw_plan_dft_3d (height, width, depth, pattern_volume, pattern_volume,
                                                    FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute (signal_forward_plan);
    fftw_execute (pattern_forward_plan);

    // cross power spectrum
    fftw_complex *cross_volume = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*size);
    double temp;
    for (int i=0; i < size; i++)
    {
        cross_volume[i][0] = (signal_volume[i][0]*pattern_volume[i][0])-
                (signal_volume[i][1]*(-pattern_volume[i][1]));
        cross_volume[i][1] = (signal_volume[i][0]*(-pattern_volume[i][1]))+
                (signal_volume[i][1]*pattern_volume[i][0]);
        temp = sqrt(cross_volume[i][0]*cross_volume[i][0]+cross_volume[i][1]*cross_volume[i][1]);
        cross_volume[i][0] /= temp;
        cross_volume[i][1] /= temp;
    }

    // backward fft
    // FFTW computes an unnormalized transform,
    // in that there is no coefficient in front of
    // the summation in the DFT.
    // In other words, applying the forward and then
    // the backward transform will multiply the input by n.

    // BUT, we only care about the maximum of the inverse DFT,
    // so we don't need to normalize the inverse result.

    // the storage order in FFTW is row-order
    fftw_plan cross_backward_plan = fftw_plan_dft_3d(height, width, depth, cross_volume, cross_volume,
                                                     FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(cross_backward_plan);

    // free memory
    fftw_destroy_plan(signal_forward_plan);
    fftw_destroy_plan(pattern_forward_plan);
    fftw_destroy_plan(cross_backward_plan);
    fftw_free(signal_volume);
    fftw_free(pattern_volume);

    Eigen::VectorXf cross_real(size);

    for (int i= 0; i < size; i++)
    {
        cross_real(i) = cross_volume[i][0];
    }

    std::ptrdiff_t max_loc;
    float unuse = cross_real.maxCoeff(&max_loc);

    height_offset =floor(((int) max_loc)/ (width*depth));
    width_offset = floor(((int)max_loc - width*depth*height_offset)/depth);
    depth_offset = floor((int)max_loc-width*depth*height_offset-width_offset*depth);

    if (height_offset > 0.5*height)
        height_offset = height_offset-height;
    if (width_offset  > 0.5*width)
        width_offset = width_offset-width;
    if (depth_offset > 0.5*depth)
        depth_offset = depth_offset-depth;
}

void voxelsize2volumesize ( const PointCloudT cloud,
                            Eigen::Vector3f voxelsize,
                            Eigen::Vector3i &volumesize)
{
    // Note that only the x,y,z field of
    // 'minpt&maxpt' make sense
    // That is downside of template, I guess
    PointT minpt, maxpt;
    pcl::getMinMax3D<PointT> (cloud, minpt, maxpt);

    volumesize(0) = ceil((maxpt.x-minpt.x)/voxelsize(0))+1;
    volumesize(1) = ceil((maxpt.y-minpt.y)/voxelsize(1))+1;
    volumesize(2) = ceil((maxpt.z-minpt.z)/voxelsize(2))+1;

    // pad the volume to be of 2^n, which FFTW favors
    if (remainder(volumesize(0),2)!=0)
        volumesize(0)++;
    if (remainder(volumesize(1),2)!=0)
        volumesize(1)++;
    if (remainder(volumesize(2),2)!=0)
        volumesize(2)++;
}

void point2volume (const PointCloudT cloud,
                   Eigen::Vector3f voxelsize,
                   Eigen::Vector3i volumesize,
                   Eigen::Vector3i volumesize_origin,
                   double *volume)
{
    int x_index, y_index, z_index;
    PointT origin_minpt, origin_maxpt;
    pcl::getMinMax3D<PointT> (cloud, origin_minpt, origin_maxpt);

    // determine the range of volume
    PointT minpt, maxpt;
    // please note that the elements of volumesize are all even,
    // so is the volumesize distance
    Eigen::Vector3i volumesize_dist = volumesize_origin - volumesize;
    minpt.x = origin_minpt.x + volumesize_dist(0)/2*voxelsize(0);
    maxpt.x = origin_maxpt.x - volumesize_dist(0)/2*voxelsize(0);
    minpt.y = origin_minpt.y + volumesize_dist(1)/2*voxelsize(1);
    maxpt.y = origin_maxpt.y - volumesize_dist(1)/2*voxelsize(1);
    minpt.z = origin_minpt.z + volumesize_dist(2)/2*voxelsize(2);
    maxpt.z = origin_maxpt.z - volumesize_dist(2)/2*voxelsize(2);

    // core part of generate volume
    for (PointCloudT::const_iterator itr=cloud.begin();
         itr!=cloud.end(); itr++)
    {
        if (    (*itr).x > minpt.x && (*itr).x < maxpt.x &&
                (*itr).y > minpt.y && (*itr).y < maxpt.y &&
                (*itr).z > minpt.z && (*itr).z < maxpt.z)
        {
            x_index = floor(((*itr).x - minpt.x)/voxelsize(0));
            y_index = floor(((*itr).y - minpt.y)/voxelsize(1));
            z_index = floor(((*itr).z - minpt.z)/voxelsize(2));

            if(isFinite(*itr))
                volume[z_index+volumesize(2)*(y_index+volumesize(1)*x_index)]++;
        }
    }
}

#endif /*TAMS_SEI_REGISTRATION_HPP_*/
