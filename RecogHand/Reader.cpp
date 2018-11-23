//
//  Reader.cpp
//  RecogHand
//
//  Created by 中山一輝 on 2018/11/23.
//  Copyright © 2018年 中山一輝. All rights reserved.
//

#include "Reader.h"

SingleImageReader::SingleImageReader(const char* path) {
    mImg = cv::imread(path);
}

bool SingleImageReader::isOpened() const {
    return mImg.data != NULL;
}

IReader& SingleImageReader::operator >> (cv::Mat& out_img) {
    out_img = mImg;
    return *this;
}

VideoReader::VideoReader(int cam) : mCapture(cam) {
    
}

bool VideoReader::isOpened() const {
    return mCapture.isOpened();
}

IReader& VideoReader::operator >> (cv::Mat& out_img) {
    mCapture >> out_img;
    return *this;
}
