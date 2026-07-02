#include <iostream>
#include <boost/filesystem.hpp>

#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

#include "opencv2/opencv.hpp"

using namespace std;

typedef pcl::PointCloud<pcl::PointXYZ> Cloud;

void listdir(const string &dir, vector<string> &output)
{
    vector<boost::filesystem::path> files;
    copy(boost::filesystem::directory_iterator(dir), boost::filesystem::directory_iterator(), back_inserter(files));
    sort(begin(files), end(files));
    for (const auto &file: files)
        output.emplace_back(file.string());
}

class Vis
{
public:
    string dataRootDir;
    pcl::visualization::PCLVisualizer visualizer;
    vector<string> pcdFiles, imgFiles;
    Cloud::Ptr cloudPtr;
    size_t size = 0;

    size_t i = 0;

    Vis() : cloudPtr(new Cloud)
    {
        dataRootDir = "/home/bruce/ROS/dev_ws/src/carla_data_collector/data/1";
        listdir(dataRootDir + "/pcd", pcdFiles);
        listdir(dataRootDir + "/img", imgFiles);
        size = pcdFiles.size();

        visualizer.setSize(640, 480);
        visualizer.setBackgroundColor(1, 1, 1);
        visualizer.addCoordinateSystem(5);
        visualizer.addPointCloud(cloudPtr);
        visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3);
        visualizer.registerKeyboardCallback(&Vis::keyboardCallback, *this);
    }

    void keyboardCallback(const pcl::visualization::KeyboardEvent &event, void *nothing)
    {
        if (event.keyDown())
        {
            if (event.getKeyCode() == '.' and i < size - 1) ++i;
            else if (event.getKeyCode() == ',' and i > 0) --i;

            pcl::io::loadPCDFile(pcdFiles[i], *cloudPtr);
            pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(cloudPtr, "z");
            visualizer.updatePointCloud(cloudPtr, fildColor);
            auto img = cv::imread(imgFiles[i]);
            cv::imshow("img", img);
            cv::waitKey(1);
        }
    }

    void spin()
    {
        visualizer.spin();
    }
};


int main(int argc, char **argv)
{
    Vis vis;
    vis.spin();

//    string dataRootDir = "/home/bruce/ROS/dev_ws/src/carla_data_collector/data/1";
//    vector<string> pcdFiles, imgFiles;
//
//    listdir(dataRootDir + "/pcd", pcdFiles);
//    listdir(dataRootDir + "/img", imgFiles);
//
//    pcl::visualization::PCLVisualizer visualizer;
//    visualizer.addCoordinateSystem(5);
//    Cloud::Ptr cloudPtr(new Cloud);
//    visualizer.addPointCloud(cloudPtr);
//
//    for (size_t i = 0; i < pcdFiles.size(); ++i)
//    {
//        pcl::io::loadPCDFile(pcdFiles[i], *cloudPtr);
//        visualizer.updatePointCloud(cloudPtr);
//        visualizer.spinOnce();
//
//        auto img = cv::imread(imgFiles[i]);
//        cv::imshow("img", img);
//        cv::waitKey();
//    }

    return 0;
}
