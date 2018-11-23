//
//  Processor.h
//  RecogHand
//
//  Created by 中山一輝 on 2018/11/23.
//  Copyright © 2018年 中山一輝. All rights reserved.
//

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class Processor {
public:
    static void recog(const cv::Mat& input, cv::Mat& output);
    
private:    
    static void swapIfBigger(std::pair<int, double>& a, std::pair<int, double>& b);
    static void extractSkin(const cv::Mat& frame, cv::Mat& dst);
    static int calcThreePointAngle(const cv::Point& a, const cv::Point& b, const cv::Point& c);
    static float calcDistance(const cv::Point& p1, const cv::Point& p2);
    static float dotProduct(const cv::Point& s1, const cv::Point& e1,
                            const cv::Point& s2, const cv::Point& e2);

    Processor() {}
    ~Processor() {}
};
