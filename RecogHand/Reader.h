//
//  Reader.h
//  RecogHand
//
//  Created by 中山一輝 on 2018/11/23.
//  Copyright © 2018年 中山一輝. All rights reserved.
//

#pragma once

#include <opencv2/opencv.hpp>

class IReader {
public:
    virtual ~IReader() {}
    virtual bool isOpened() const = 0;
    virtual IReader& operator >> (cv::Mat& out_img) = 0;
};

class SingleImageReader : public IReader {
public:
    SingleImageReader(const char* path);
    bool isOpened() const;
    IReader& operator >> (cv::Mat& out_img);

private:
    cv::Mat mImg;
};

class VideoReader : public IReader {
public:
    VideoReader(int cam);
    bool isOpened() const;
    IReader& operator >> (cv::Mat& out_img);

private:
    cv::VideoCapture mCapture;
};
