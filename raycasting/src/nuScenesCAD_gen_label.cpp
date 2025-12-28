#include <iostream>
#include <filesystem>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "Eigen/Eigen"
#include "utils.hpp"

// PCL related
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

using namespace std;
namespace fs = std::filesystem;
typedef fs::path Path;

struct NuScenesPoint {
    float x, y, z, intensity, ring;
};

void convertToPCD(const Path &inputFile, const Path &outputFile) {
    ifstream ifs(inputFile, ios::binary);
    if (!ifs.is_open())
    {
        cout << "    Failed to open point cloud file: " << inputFile.filename() << ", skipping..." << endl;
        return;
    }
    vector<NuScenesPoint> rawPoints;
    NuScenesPoint point;
    while (ifs.read(reinterpret_cast<char*>(&point), sizeof(NuScenesPoint)))
    {
        rawPoints.push_back(point);
    }
    ifs.close();


    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    cloud->reserve(rawPoints.size());
    for (const auto& p : rawPoints)
    {
        pcl::PointXYZI pt;
        pt.x = p.x;
        pt.y = p.y;
        pt.z = p.z;
        pt.intensity = p.intensity;
        cloud->push_back(pt);
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;
    pcl::io::savePCDFileBinary(outputFile.string(), *cloud);
    cout << "    Converted point cloud to: " << outputFile.filename() << endl;
}


Eigen::Vector2d voxelTraversal(
        const cv::Mat &img,
        const Eigen::Vector2d &start,
        const Eigen::Vector2d &ray,
        const map<string, cv::Vec3b> &label_bgr
)
{
    Eigen::Vector2i currentVoxel(int(ceil(start[0])), int(ceil(start[1])));

    auto &vx = currentVoxel[0], &vy = currentVoxel[1];
    const auto vxSize = img.cols, vySize = img.rows;

    auto stepX = (ray[0] >= 0) ? 1 : -1;
    auto stepY = (ray[1] >= 0) ? 1 : -1;

    auto tMaxX = (ray[0] != 0) ? double(stepX) / ray[0] : numeric_limits<float>::max();
    auto tMaxY = (ray[1] != 0) ? double(stepY) / ray[1] : numeric_limits<float>::max();

    auto tDeltaX = tMaxX;
    auto tDeltaY = tMaxY;

    Eigen::Vector2i diff(0, 0);
    bool negRay = false;
    if (ray[0] < 0)
    {
        --diff[0];
        negRay = true;
    }
    if (ray[1] < 0)
    {
        --diff[1];
        negRay = true;
    }

    if (negRay) currentVoxel += diff;

    int cntX = 0, cntY = 0, through;
    while (true)
    {
        if (tMaxX < tMaxY)
        {
            ++cntX;
            auto nextX_ = vx + stepX;
            if (nextX_ < 0 or
                nextX_ >= vxSize or
                (img.at<cv::Vec3b>(img.rows - 1 - vy, nextX_) != label_bgr.at("road") and img.at<cv::Vec3b>(img.rows - 1 - vy, nextX_) != label_bgr.at("normal")))
            {
                through = 1;  /// Hit vertical border
                break;
            }
            vx = nextX_;
            tMaxX += tDeltaX;
        }
        else
        {
            ++cntY;
            auto nextY_ = vy + stepY;
            if (nextY_ < 0 or
                nextY_ >= vySize or
                (img.at<cv::Vec3b>(img.rows - 1 - nextY_, vx) != label_bgr.at("road") and img.at<cv::Vec3b>(img.rows - 1 - nextY_, vx) != label_bgr.at("normal")))
            {
                through = 0;  /// Hit horizontal border
                break;
            }
            vy = nextY_;
            tMaxY += tDeltaY;
        }
    }

    double endX, endY;
    if (through == 0)
    {
        auto deltaY = cntY * stepY;
        endY = deltaY + start[1];
        endX = ray[0] / ray[1] * deltaY + start[0];
    }
    else
    {
        auto deltaX = cntX * stepX;
        endX = deltaX + start[0];
        endY = ray[1] / ray[0] * deltaX + start[1];
    }

    return {endX, endY};
}

void calEnd(
        const cv::Mat &img,
        const Eigen::Vector2d &start,
        vector<Eigen::Vector2d> &ends,
        double minAngle,
        const map<string, cv::Vec3b> &label_bgr
)
{
    double angleDiff = 2 * M_PI / double(ends.size());
    auto startAngle = minAngle + angleDiff / 2;
    for (auto i = 0; i < ends.size(); ++i)
    {
        auto theta = startAngle + i * angleDiff;
        Eigen::Vector2d ray(cos(theta), sin(theta));
        ends[i] = voxelTraversal(img, start, ray, label_bgr);
    }
}

shared_ptr<double[]> cvtEndToLaser(
        const Eigen::Vector2d &start,
        const vector<Eigen::Vector2d> &ends,
        double resolution
)
{
    shared_ptr<double[]> laser(new double[ends.size()]);
    for (auto i = 0; i < ends.size(); ++i) {
        double temp = (ends[i] - start).norm() * resolution;
        // we filter out the max range
        if (temp > 20.0)
            laser[i] = 20;
        else laser[i] = temp;
    }

    return laser;
}

cv::Mat visualizeLaser(
        const cv::Mat &src,
        const Eigen::Vector2d &start,
        const vector<Eigen::Vector2d> &ends
)
{
    auto img = src.clone();

//    size_t cnt = 0;
    for (auto &end: ends)
    {
        cv::line(
                img,
                cv::Point( int(start[0]), img.rows - 1 - int(start[1])),
                cv::Point(int(end[0]), img.rows - 1 - int(end[1])),
                cv::Scalar(128, 90, 128)
        );
        img.at<cv::Vec3b>(img.rows - 1 - int(end[1]), int(end[0])) = cv::Vec3b(0, 0, 255);
//        if (cnt % 10 == 0)
//            cv::putText(
//                    img,
//                    to_string(cnt),
//                    cv::Point( end[0], img.rows - 1 - end[1]),
//                    cv::FONT_HERSHEY_SIMPLEX,
//                    0.25,
//                    cv::Scalar(0, 255, 255)
//            );
//        ++cnt;
    }

    return img;
}

int main(int argc, char **argv)
{
    // nuScenes type. road means driveable surface.
    const map<string, cv::Vec3b> label_bgr = {
            {"vehicle",  cv::Vec3b(142, 0, 0)},
            {"road",     cv::Vec3b(191, 207, 0)},
            {"normal",   cv::Vec3b(255, 255, 255)},
            {"lane",     cv::Vec3b(50, 234, 157)},
            {"lamp",     cv::Vec3b(153, 153, 153)},
            {"sidewalk", cv::Vec3b(232, 35, 244)}
    };

    Path dataDir = "your-nuScenes-CAD-path/"; //
    size_t laserNum = 384;
    double resolution = 40. / 200;  // meter / pixel

    for (const auto &sceneEntry : fs::directory_iterator(dataDir)) {
        if (!sceneEntry.is_directory() || sceneEntry.path().filename().string().find("scene") != 0) {
            continue;
        }
        Path sceneDir = sceneEntry.path();
        cout << "Processing scene " << sceneDir.filename().string() << endl;

        // Define input and output directories
        Path segImgDir = sceneDir / "seg_img";
        Path labelDir = sceneDir / "label";
        Path laserDir = sceneDir / "laser";
        Path lidarDir = sceneDir / "lidar";
        Path pcdDir = sceneDir / "pcd";

        fs::create_directories(labelDir);
        fs::create_directories(laserDir);
        fs::create_directories(pcdDir);

        if (!fs::exists(segImgDir) || !fs::is_directory(segImgDir)) {
            cout << "seg_img directory not found in " << sceneDir.filename().string() << ", skipping ..." << endl;
            continue;
        }
        if (!fs::exists(lidarDir) || !fs::is_directory(lidarDir))
        {
            cout << "lidar directory not found in " << sceneDir.filename().string() << ", skipping..." << endl;
            continue;
        }

        // Collect image files from seg_img
        vector<pair<Path, Path>> files;
        for (const auto &imgEntry : fs::directory_iterator(segImgDir))
        {
            if (imgEntry.path().extension() != ".jpg" && imgEntry.path().extension() != ".png") {
                cout << "Skipping non-image file: " << imgEntry.path().filename() << endl;
                continue;
            }

            auto fileID = imgEntry.path().filename().string().substr(0, 6);
            Path lidarFile = lidarDir / (fileID + ".pcd.bin");
            if (fs::exists(lidarFile))
            {
                cout << "Matching: " << imgEntry.path().filename() << " with " << lidarFile.filename() << endl;
                files.emplace_back(std::make_pair(imgEntry.path(), lidarFile));
            }
            else {
                cout << "No matching lidar file for: " << imgEntry.path().filename() << ", skipping..." << endl;
            }
        }

        if (files.empty()) {
            cout << "  No valid image-lidar pairs found in " << sceneDir.filename().string() << ", skipping..." << endl;
            continue;
        }

        for (const auto &[imgFile, lidarFile]: files) {
            auto fileName = imgFile.filename();
            cout << "Processing image : " << fileName << endl;
            auto fileID = fileName.string().substr(0, 6);
            auto img = cv::imread(imgFile.string());

            if (img.empty()) {
                cout << "Failed to load image: " << fileName << ", skipping..." << endl;
                continue;
            }
            // Processing sematic BEV image into laser & segimg
            Eigen::Vector2d origin(double(img.cols) / 2 - 0.5, double(img.rows) / 2 - 0.5);
            vector<Eigen::Vector2d> ends(laserNum);
            calEnd(img, origin, ends, -M_PI, label_bgr);

            auto vis = visualizeLaser(img, origin, ends);
            cv::imwrite((laserDir / (fileID + ".png")).string(), vis);

            auto laserPtr = cvtEndToLaser(origin, ends, resolution);
            ofstream ofs((labelDir / (fileID + ".data")).string(), ios::binary);
            ofs.write((char *) &(laserPtr[0]), sizeof(double) * laserNum);
            ofs.close();

            convertToPCD(lidarFile, pcdDir / (fileID + ".pcd"));
        }
    }
    cout << "Processing completed !" << endl;
    return 0;
}
