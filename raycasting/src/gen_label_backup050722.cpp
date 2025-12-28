//
// Created by bruce on 2021/11/18.
//

#include <iostream>
#include <filesystem>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "Eigen/Eigen"
#include "utils.hpp"

using namespace std;
typedef filesystem::path Path;


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
                img.at<cv::Vec3b>(img.rows - 1 - vy, nextX_) != label_bgr.at("road"))
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
                img.at<cv::Vec3b>(img.rows - 1 - nextY_, vx) != label_bgr.at("road"))
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
    for (auto i = 0; i < ends.size(); ++i)
        laser[i] = (ends[i] - start).norm() * resolution;
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
        img.at<cv::Vec3b>(img.rows - 1 - int(end[1]), int(end[0])) = cv::Vec3b(0, 255, 255);
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
    const map<string, cv::Vec3b> label_bgr = {
            {"vehicle",  cv::Vec3b(142, 0, 0)},
            {"road",     cv::Vec3b(128, 64, 128)},
            {"lane",     cv::Vec3b(50, 234, 157)},
            {"lamp",     cv::Vec3b(153, 153, 153)},
            {"sidewalk", cv::Vec3b(232, 35, 244)}
    };

    Path dataDir = "/home/vsisauto/temp/cpp/carla_data/data/town02";
    vector<Path> imgFiles;
    listdir(dataDir / "filtered", imgFiles);

    size_t laserNum = 384;
    double resolution = 30. / 600;  // meter / pixel

    for (auto &imgFile: imgFiles)
    {
        auto fileName = imgFile.filename();
        cout << fileName << endl;
        auto fileID = fileName.string().substr(0, 6);
        auto img = cv::imread(imgFile);

        Eigen::Vector2d origin(double(img.cols) / 2 - 0.5, double(img.rows) / 2 - 0.5);
        vector<Eigen::Vector2d> ends(laserNum);
        calEnd(img, origin, ends, -M_PI, label_bgr);

        auto vis = visualizeLaser(img, origin, ends);
//        cv::imshow("laser", vis);
//        cv::waitKey();
        cv::imwrite(dataDir / "laser" / (fileID + ".png"), vis);

        auto laserPtr = cvtEndToLaser(origin, ends, resolution);
        ofstream ofs(dataDir / "label" / (fileID + ".data"), ios::binary);
        ofs.write((char *) &(laserPtr[0]), sizeof(double) * laserNum);
        ofs.close();
    }

    return 0;
}
