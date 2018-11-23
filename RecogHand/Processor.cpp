//
//  Processor.cpp
//  RecogHand
//
//  Created by 中山一輝 on 2018/11/23.
//  Copyright © 2018年 中山一輝. All rights reserved.
//

#include "Processor.h"

#include <opencv2/objdetect/objdetect.hpp>
#include <math.h>
#include <vector>

// MACROS
#define rad_to_deg(rad) (((rad)/2/M_PI)*360)

// CONST
static const float MIN_FINGER_TOP_LENGTH = 30;
static const float MAX_FINGER_WIDTH = 100;

// STRUCT
struct Finger {
    cv::Point root;
    cv::Point top;
    
    Finger() {}
    Finger(const cv::Point& r, const cv::Point& t) : root(r), top(t) {}
};

void Processor::swapIfBigger(std::pair<int, double>& a, std::pair<int, double>& b) {
    if (a.second < b.second) {
        std::swap(a, b);
    }
}

void Processor::extractSkin(const cv::Mat& frame, cv::Mat& dst) {
    cv::Mat mask;
    cv::Mat hsv;
    
    // HSVへの変換
    cv::cvtColor(frame, hsv, CV_BGR2HSV);
    
    // 肌色の箇所を抽出してマスク画像を作成する
    cv::inRange(hsv, cv::Scalar(0, 70, 90), cv::Scalar(32, 255, 255), mask);
    
    // マスク処理で使用する入力画像の作成
    cv::Mat whiteImage = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
    whiteImage = cv::Scalar(255);
    
    // 肌色の部分のみ255にする
    cv::bitwise_and(whiteImage, whiteImage, dst, mask);
}

int Processor::calcThreePointAngle(const cv::Point& a, const cv::Point& b, const cv::Point& c) {
    float rad1 = 2 * M_PI + atan2(b.y - a.y, a.x - b.x);
    float rad2 = 2 * M_PI + atan2(b.y - c.y, c.x - b.x);
    int angle rad_to_deg(rad1 - rad2);
    return angle % 360;
}

float Processor::calcDistance(const cv::Point& p1, const cv::Point& p2) {
    float dist = p1.x - p2.x;
    float dist2 = dist * dist;
    dist = p1.y - p2.y;
    dist2 += dist * dist;
    return sqrt(dist2);
}

float Processor::dotProduct(const cv::Point& s1, const cv::Point& e1,
                        const cv::Point& s2, const cv::Point& e2) {

    float vec1X = s1.x - e1.x;
    float vec1Y = s1.y - e1.y;
    float n = sqrt(vec1X * vec1X + vec1Y * vec1Y);
    vec1X /= n;
    vec1Y /= n;
    
    float vec2X = s2.x - e2.x;
    float vec2Y = s2.y - e2.y;
    n = sqrt(vec2X * vec2X + vec2Y * vec2Y);
    vec2X /= n;
    vec2Y /= n;

    return vec1X * vec2X + vec1Y * vec2Y;
}

void Processor::recog(const cv::Mat& input, cv::Mat& output) {
    cv::Mat hand_bin; // 手の二値画像
    extractSkin(input, hand_bin);
    
    auto kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
    
    // 手の周囲の雑音を消す
    cv::morphologyEx(hand_bin, hand_bin, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 5);
    
    // 手の中の雑音を消す
    cv::morphologyEx(hand_bin, hand_bin, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 5);
    
    cv::Mat hand_bin_tmp = hand_bin.clone();
    
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(hand_bin_tmp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    hand_bin_tmp.release();
    
    output = input.clone();
    
    // 面積の大きい上位２つの輪郭を選ぶ
    std::pair<int, double> areaRanking[2] = {{-1,0}, {-1, 0}};
    for (int i = 0; i < contours.size(); i++){
        const std::vector<cv::Point>& points = contours[i];
        
        double area = cv::contourArea(points);
        std::pair<int, double> value(i, area);
        
        swapIfBigger(areaRanking[0], value);
        swapIfBigger(areaRanking[1], value);
    }
    
    // 輪郭線の描画
    //cv::drawContours(output, contours, areaRanking[0].first, cv::Scalar(255, 0, 0), 2);
    //cv::drawContours(output, contours, areaRanking[1].first, cv::Scalar(255, 255, 0), 2);
    
    // デバック情報の描画
    const int span = 100;
    cv::line(output, cv::Point(0, 0), cv::Point(span, 0), cv::Scalar(255, 0, 0), 2);

    // 面積の上位２つのうち、どちらが手かを判別する
    std::pair<int, size_t> handIndex(-1, 0);
    for (int i = 0; i < 2 && areaRanking[i].first >= 0; i++) {
        const std::vector<cv::Point>& points = contours[areaRanking[i].first];

        int numPoints = (int)points.size();
        std::vector<Finger> fingers;
        for (int start = 0; start < numPoints; start++) {
            int cindex = (start + (span / 2)) % points.size();
            int eindex = (start + span) % points.size();
            
            cv::Point s = points[start];
            cv::Point c = points[cindex];
            cv::Point e = points[eindex];
            
            if (MIN_FINGER_TOP_LENGTH < calcDistance(s, c) &&
                    MIN_FINGER_TOP_LENGTH < calcDistance(e, c)) {
                float angle = calcThreePointAngle(s, c, e);
                if (30 < angle && angle < 90) {
                    
                    int straight_span = span / 2;
                    
                    // s からマイナス方向に直線が続いているかを判定する
                    bool straight_s = false;
                    int ss = start;
                    int sc = ss - (straight_span / 2);
                    if (sc < 0) {
                        sc = numPoints + (sc % numPoints);
                    }

                    int se = ss - straight_span;
                    if (se < 0) {
                        se = numPoints + (se % numPoints);
                    }
                    int angle = std::abs(calcThreePointAngle(points[sc], points[ss], points[se]));
                    if (0 <= angle && angle < 20) {
                        straight_s = true;
                    }
                    
                    // e からプラス方向に直線が続いているかを判定する
                    bool straight_e = false;
                    int es = eindex;
                    int ec = (es + (straight_span / 2)) % numPoints;
                    int ee = (es + straight_span) % numPoints;
                    int angle2 = std::abs(calcThreePointAngle(points[ec], points[es], points[ee]));
                    if (0 <= angle2 && angle2 < 20) {
                        straight_e = true;
                    }
                    
                    if (straight_s && straight_e) {
                        float dot = dotProduct(points[ss], points[se], points[es], points[ee]);
                        if (dot > 0.85) {
                            
                            float topLength = calcDistance(points[ss], points[es]);
                            if (topLength < MAX_FINGER_WIDTH) {
                                float bottomLength = calcDistance(points[se], points[ee]);
                                if (bottomLength < MAX_FINGER_WIDTH) {
                                    cv::circle(output, c, 10, cv::Scalar(0, 255, 0), -1);
                                    cv::line(output, points[ss], points[se], cv::Scalar(0, 255, 255), 2);
                                    cv::line(output, points[es], points[ee], cv::Scalar(0, 255, 255), 2);
                                    printf("angle(%d, %d) rad=%f\n", angle, angle2, dot);
                                    
                                    cv::circle(output, points[ss], 10, cv::Scalar(255, 255, 0), -1);
                                    cv::circle(output, points[es], 10, cv::Scalar(255, 255, 0), -1);
                                    cv::circle(output, points[se], 10, cv::Scalar(255, 255, 255), -1);
                                    cv::circle(output, points[ee], 10, cv::Scalar(255, 255, 255), -1);
                                    
                                    Finger finger;
                                    finger.top = c;
                                    finger.root = points[se] + points[ee];
                                    finger.root.x /= 2;
                                    finger.root.y /= 2;
                                    fingers.push_back(finger);
                                    
                                    cv::line(output, finger.top, finger.root, cv::Scalar(0, 0, 255), 10);
                                    
                                    start += span - 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if (0 < fingers.size() && ((handIndex.first < 0) || (handIndex.second < fingers.size()))) {
            handIndex.first = i;
            handIndex.second = fingers.size();
        }
    }
    
    // 手にフォーカスを当てる(バウンディングボックスを描画する)
    if (handIndex.first >= 0) {
        int cIndex = areaRanking[handIndex.first].first;
        cv::Rect brect = cv::boundingRect(contours[cIndex]);
        cv::rectangle(output, brect.tl(), brect.br(), cv::Scalar(0, 0, 255), 2);
    }
    
    cv::cvtColor(hand_bin, hand_bin, CV_GRAY2BGR);
    cv::resize(hand_bin, hand_bin, cv::Size(), 0.2, 0.2);
    cv::Rect roi_rect(output.cols - hand_bin.cols, output.rows - hand_bin.rows, hand_bin.cols, hand_bin.rows);
    cv::Mat roi = output(roi_rect);
    hand_bin.copyTo(roi);
    
}

