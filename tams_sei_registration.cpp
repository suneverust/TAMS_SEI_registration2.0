/*********************************************
 * Author: Bo Sun                            *
 * Afflication: TAMS, University of Hamburg  *
 * E-Mail: bosun@informatik.uni-hamburg.de   *
 *         user_mail@QQ.com                  *
 * Date: Nov 13, 2014                        *
 * Licensing: GNU GPL license.               *
 *********************************************/


#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>

#include <vector>
#include <math.h>
#include <algorithm>
#include <eigen3/Eigen/Dense>

#include "tams_soft_fftw_correlate.h"
#include "tams_sei_registration.h"
#include "tams_sei_registration.hpp"

// For debug
#include <time.h>
#include <fstream>

#ifndef TYPE_DEFINITION_
#define TYPE_DEFINITION_
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
// Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;
#endif /*TYPE_DEFINITION_*/

std::string object_filename_;
std::string scene_filename_;
int sei_dim = 32;
int hist_bin = 12;
Eigen::Vector3f voxel_size;

void showhelp(char *filename)
{
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "-                                                                      -" << std::endl;
    std::cout << "-               TAMS_SEI_REGISTRATION2.0  User Guide                   -" << std::endl;
    std::cout << "-                                                                      -" << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << filename << " object_filename.pcd scene_filename.pcd [Options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "      -h               Show this help." << std::endl;
    std::cout << "      --sei_dim        Set the dimension of SEI" << std::endl;
    std::cout << "      --hist_bin       Set the dimension of histogram in entropy computation" << std::endl;
    std::cout << "      --voxel_size     Set the size of voxel when render point cloud into 3D volume" << std::endl;
    std::cout << std::endl;
}

void parseCommandLine(int argc, char **argv)
{
    if (argc < 3)
    {
        showhelp(argv[0]);
        exit(-1);
    }

    if (pcl::console::find_switch(argc, argv, "-h"))
    {
        showhelp(argv[0]);
        exit(0);
    }

    std::vector<int> filenames;
    filenames = pcl::console::parse_file_extension_argument(argc, argv, ".pcd");
    if (filenames.size()!=2)
    {
        std::cout << "\nFilenames missing.\n";
        showhelp(argv[0]);
        exit(-1);
    }
    object_filename_ = argv[filenames[0]];
    scene_filename_ = argv[filenames[1]];

    pcl::console::parse_argument(argc, argv, "--sei_dim", sei_dim);
    pcl::console::parse_argument(argc, argv, "--hist_bin", hist_bin);
    if(pcl::console::parse_3x_arguments(
                argc, argv, "--voxel_size", voxel_size(0), voxel_size(1), voxel_size(2))
            ==-1)
    {
        voxel_size << 0.5, 0.5, 0.5;
    }
}

int
main (int argc, char**argv)
{
    parseCommandLine(argc, argv);

    // Point clouds
    PointCloudT::Ptr object(new PointCloudT);
    PointCloudT::Ptr object_norm(new PointCloudT);
    PointCloudT::Ptr scene(new PointCloudT);
    PointCloudT::Ptr scene_norm(new PointCloudT);
    PointCloudT::Ptr object_trans (new PointCloudT);
    PointCloudT::Ptr object_final(new PointCloudT);

    // Load object and scene point clouds
    if (pcl::io::loadPCDFile<PointT>(object_filename_, *object)<0 ||
            pcl::io::loadPCDFile<PointT>(scene_filename_,*scene)<0)
    {
        pcl::console::print_error("Error loading object/scene file!\n");
        return (1);
    }

    // Remove the NaN points in object and scene if any
    pcl::console::print_highlight("Remove the NaN points if any...\n");
    std::vector<int> indices_object_nan, indices_scene_nan;
    pcl::removeNaNFromPointCloud(*object,*object,indices_object_nan);
    pcl::removeNaNFromPointCloud(*scene, *scene, indices_scene_nan);

    // translation normalization
    Eigen::Vector4f object_centroid;
    Eigen::Vector4f scene_centroid;
    pcl::compute3DCentroid(*object, object_centroid);
    pcl::compute3DCentroid(*scene, scene_centroid);

    Eigen::Affine3f object_norm_transform = Eigen::Affine3f::Identity();
    object_norm_transform.translation() << object_centroid[0], object_centroid[1], object_centroid[2];
    Eigen::Affine3f scene_norm_transform = Eigen::Affine3f::Identity();
    scene_norm_transform.translation() << scene_centroid[0], scene_centroid[1], scene_centroid[2];

    pcl::transformPointCloud(*object, *object_norm, object_norm_transform);
    pcl::transformPointCloud(*scene, *scene_norm, scene_norm_transform);

    // generate SEI
    // Damn Lucky! Negative and Negative equal Positive!
    pcl::console::print_highlight("Generate the SEIs...\n");
    Eigen::MatrixXf sei_object = Eigen::MatrixXf::Zero(2*sei_dim, 2*sei_dim);
    Eigen::MatrixXf sei_scene  = Eigen::MatrixXf::Zero(2*sei_dim, 2*sei_dim);

    computeSEI(*object_norm, sei_dim, hist_bin, sei_object);
    computeSEI(*scene_norm,  sei_dim, hist_bin, sei_scene);

    // free memory
    (*object_norm).clear();
    (*scene_norm).clear();

    // sei resize
    Eigen::VectorXf sei_object_vector = Eigen::VectorXf::Zero(2*sei_dim*2*sei_dim);
    Eigen::VectorXf sei_scene_vector  = Eigen::VectorXf::Zero(2*sei_dim*2*sei_dim);

    sei_object.resize(2*sei_dim*2*sei_dim,1);
    sei_scene.resize(2*sei_dim*2*sei_dim,1);

    sei_object_vector << sei_object;
    sei_scene_vector << sei_scene;

    if (sei_object_vector.size()!= 2*sei_dim*2*sei_dim ||
            sei_scene_vector.size()!=2*sei_dim*2*sei_dim)
    {
        pcl::console::print_error("SEI resize Failed!\n");
        return (1);
    }

    // free memory
    sei_object.resize(0,0);
    sei_scene.resize(0,0);

    // rotation recovery
    pcl::console::print_highlight("Rotation recovery...\n");
    int bwIn = sei_dim;
    int bwOut = sei_dim;
    int digLim = sei_dim-1;

    double alpha , beta, gamma;
    tams_soft_fftw_correlate(sei_scene_vector,
                             sei_object_vector,
                             bwIn, bwOut, digLim,
                             alpha, beta, gamma);
    pcl::console::print_info("Rotation results: alpha = %f, beta = %f, gamma = %f\n", alpha, beta, gamma);

    // generate the rotation matrix
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.prerotate(Eigen::AngleAxisf(gamma,Eigen::Vector3f::UnitZ()));
    transform.prerotate(Eigen::AngleAxisf(beta,Eigen::Vector3f::UnitY()));
    transform.prerotate(Eigen::AngleAxisf(alpha,Eigen::Vector3f::UnitZ()));

    // print rotation matrix
    printf ("Rotation estimated by global method: \n");
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transform (0,0), transform (0,1), transform (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transform (1,0), transform (1,1), transform (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transform (2,0), transform (2,1), transform (2,2));
    pcl::console::print_info ("\n");

    pcl::transformPointCloud(*object, *object_trans, transform);

    // estimate the translation
    pcl::console::print_highlight("Estimate the translation...\n");
    Eigen::Vector3i object_volume_size;
    Eigen::Vector3i scene_volume_size;
    Eigen::Vector3i volume_size;
    voxelsize2volumesize(*object_trans, voxel_size, object_volume_size);
    voxelsize2volumesize(*scene,  voxel_size, scene_volume_size);
    volume_size(0) = std::min(object_volume_size(0), scene_volume_size(0));
    volume_size(1) = std::min(object_volume_size(1), scene_volume_size(1));
    volume_size(2) = std::min(object_volume_size(2), scene_volume_size(2));

    double *object_volume;
    double *scene_volume;
    object_volume = (double*) calloc (volume_size(0)*volume_size(1)*volume_size(2),
                                      sizeof(double));
    scene_volume = (double*) calloc (volume_size(0)*volume_size(1)*volume_size(2),
                                     sizeof(double));

    point2volume(*object_trans,voxel_size,volume_size,
                 object_volume_size,object_volume);
    point2volume(*scene,voxel_size, volume_size,
                 scene_volume_size, scene_volume);
    Eigen::Vector3i offset;
    PhaseCorrelation3D(scene_volume, object_volume,
                       volume_size(0), volume_size(1), volume_size(2),
                       offset(0), offset(1), offset(2));
    transform.translation() << offset(0)*voxel_size(0),
            offset(1)*voxel_size(1),
            offset(2)*voxel_size(2);

    pcl::console::print_info ("Translation estimated by global method: \n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transform (0,3), transform (1,3), transform (2,3));
    pcl::console::print_info ("\n");

    // Refine
    pcl::IterativeClosestPoint<PointT,PointT> icp;
    icp.setInputSource(object_trans);
    icp.setInputTarget(scene);
    icp.setMaxCorrespondenceDistance(0.2);
    icp.setTransformationEpsilon(1E-04);
    icp.align(*object_final);

    std::cout << "ICP converged score:" << icp.getFitnessScore() << std::endl;
    pcl::console::print_info("\n");
    pcl::console::print_info("The Final transformation matrix:\n");
    std::cout << icp.getFinalTransformation() << std::endl;


    return (0);
}

