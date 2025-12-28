//
// Created by bruce on 2021/12/12.
//

#include "opencv2/opencv.hpp"
#include "utils.hpp"

using filesystem::path;

template<typename T>
bool in(T &value, vector<T> &vec)
{
    if (any_of(vec.begin(), vec.end(), [value](auto i) { return value == i; }))
        return true;
    return false;
}

template<typename T>
int index(T &value, vector<T> &vec)
{
    for (int i = 0; i < vec.size(); ++i)
    {
        if (value == vec[i]) return i;
    }
    return -1;
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

cv::Mat processImg(const cv::Mat &img_, map<string, cv::Vec3b> &label_bgr, const cv::Mat &egoMask)
{
    auto img = img_.clone();

    img.setTo(label_bgr["road"], egoMask);

    cv::Mat box1Mask, box2Mask, riderMask, vehicleMask;
    cv::inRange(img, label_bgr["box1"], label_bgr["box1"], box1Mask);
    cv::inRange(img, label_bgr["box2"], label_bgr["box2"], box2Mask);
    cv::inRange(img, label_bgr["rider"], label_bgr["rider"], riderMask);
    cv::inRange(img, label_bgr["vehicle"], label_bgr["vehicle"], vehicleMask);
    auto element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(box1Mask, box1Mask, cv::MORPH_CLOSE, element);
    cv::morphologyEx(box2Mask, box2Mask, cv::MORPH_CLOSE, element);

    cv::Mat laneMask;
    cv::inRange(img_, label_bgr["lane"], label_bgr["lane"], laneMask);
    img.setTo(label_bgr["road"], laneMask);

    cv::Mat signalMask;
    cv::inRange(img, label_bgr["signal"], label_bgr["signal"], signalMask);
    img.setTo(label_bgr["lamp"], signalMask);

    cv::Mat roadMask, medianMask, sidewalkMask;
    cv::inRange(img, label_bgr["road"], label_bgr["road"], roadMask);
    cv::inRange(img, label_bgr["median"], label_bgr["median"], medianMask);
    cv::inRange(img, label_bgr["sidewalk"], label_bgr["sidewalk"], sidewalkMask);

    cv::morphologyEx(roadMask, roadMask, cv::MORPH_CLOSE, element);
    cv::morphologyEx(medianMask, medianMask, cv::MORPH_CLOSE, element);
    cv::morphologyEx(sidewalkMask, sidewalkMask, cv::MORPH_CLOSE, element);

    img.setTo(label_bgr["road"], roadMask);
    img.setTo(label_bgr["median"], medianMask);
    img.setTo(label_bgr["sidewalk"], sidewalkMask);

    img.setTo(label_bgr["box1"], box1Mask);
    img.setTo(label_bgr["box2"], box2Mask);
    img.setTo(label_bgr["rider"], riderMask);
    img.setTo(label_bgr["vehicle"], vehicleMask);

    cv::Mat lampMask;
    cv::inRange(img, label_bgr["lamp"], label_bgr["lamp"], lampMask);
    img.setTo(label_bgr["road"], lampMask);

//    cv::imshow("src", img_);
//    cv::imshow("img", img);
//    cv::waitKey();
    return img;
}

cv::Mat calEgoMask(const cv::Mat &src, map<string, cv::Vec3b> &label_bgr)
{
    cv::Mat mask;
//    cv::cvtColor(cv::Mat(src == label_bgr["road"]), mask, cv::COLOR_BGR2GRAY);
//    mask.setTo(255, mask > 0);
//    cv::bitwise_not(mask, mask);
    cv::inRange(src, label_bgr["vehicle"], label_bgr["vehicle"], mask);

    cv::Mat labels;
    cv::connectedComponents(mask, labels);
    mask = labels == labels.at<int>(src.rows / 2, src.cols / 2);

    auto res = mask.clone();
    for (auto i = 0; i < mask.rows; ++i)
    {
        for (auto j = 0; j < mask.cols; ++j)
        {
            auto p = mask.at<uchar>(i, j);
            if (p == 255)
            {
                for (int i_ = -1; i_ <= 1; ++i_)
                {
                    for (int j_ = -1; j_ <= 1; ++j_)
                    {
                        res.at<uchar>(i + i_, j + j_) = 255;
                    }
                }
            }
        }
    }

    return res;
}

int main(int argc, char **argv)
{
    map<string, cv::Vec3b> label_bgr = {
            {"vehicle",  cv::Vec3b(142, 0, 0)},
            {"road",     cv::Vec3b(128, 64, 128)},
            {"lane",     cv::Vec3b(50, 234, 157)},
            {"lamp",     cv::Vec3b(153, 153, 153)},
            {"sidewalk", cv::Vec3b(232, 35, 244)},
            {"signal",   cv::Vec3b(30, 170, 250)},
            {"median",   cv::Vec3b(81, 0, 81)},
            {"box1",   cv::Vec3b(50, 120, 170)},
            {"box2",   cv::Vec3b(160, 190, 110)},
            {"rider",   cv::Vec3b(60, 20, 220)}
    };
    path imgDir = "/home/vsisauto/temp/dev_ws/src/carla_data_collector/data/town02/img";
    path savDir = "/home/vsisauto/temp/cpp/carla_data/data/town02/filtered";
    vector<path> imgFiles;
    listdir(imgDir, imgFiles);

    auto egoMask = calEgoMask(cv::imread(imgFiles[0]), label_bgr);

    for (auto &imgFile: imgFiles)
    {
        auto fileName = imgFile.filename();
        auto img = cv::imread(imgFile);
        auto filtered = processImg(img, label_bgr, egoMask);
        cv::imwrite(savDir / fileName, filtered);
        cout << fileName << endl;
//        cv::imshow("img", img);
//        cv::imshow("filtered", filtered);
//        cv::waitKey();
    }

    return 0;
}
