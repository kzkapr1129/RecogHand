//
//  main.cpp
//  RecogHand
//
//  Created by 中山一輝 on 2018/11/19.
//  Copyright © 2018年 中山一輝. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <math.h>

void swapIfBigger(std::pair<int, double>& a, std::pair<int, double>& b) {
    if (a.second < b.second) {
       std::swap(a, b);
    }
}

const std::pair<int, double>& min(const std::pair<int, double>& a, const std::pair<int, double>& b) {
    if (a.second < b.second) {
        return a;
    } else {
        return b;
    }
}

static void extractSkin(const cv::Mat& frame, cv::Mat& dst) {
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

static float calcThreePointAngle(const cv::Point& a, const cv::Point& b, const cv::Point& c) {
#define rad_to_deg(rad) (((rad)/2/M_PI)*360)
    
    float rad1 = 2 * M_PI + atan2(b.y - a.y, a.x - b.x);
    float rad2 = 2 * M_PI + atan2(b.y - c.y, c.x - b.x);
    return rad_to_deg(rad1 - rad2);
}

void doProcess(const cv::Mat& frame, cv::Mat& rgb) {
    cv::Mat hand_bin; // 手の二値画像
    extractSkin(frame, hand_bin);
    
    auto kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
    
    // 手の周囲の雑音を消す
    cv::morphologyEx(hand_bin, hand_bin, cv::MORPH_OPEN, kernel);
    
    // 手の中の雑音を消す
    cv::morphologyEx(hand_bin, hand_bin, cv::MORPH_CLOSE, kernel);
    
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(hand_bin, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    rgb = frame.clone();
    
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
    cv::drawContours(rgb, contours, areaRanking[0].first, cv::Scalar(255, 0, 0), 2);
    cv::drawContours(rgb, contours, areaRanking[1].first, cv::Scalar(255, 255, 0), 2);

    // 面積の上位２つのうち、どちらが手かを判別する
    const int span = 100;
    for (int i = 0; i < 2 && areaRanking[i].first >= 0; i++) {
        const std::vector<cv::Point>& points = contours[areaRanking[i].first];
        
        for (int start = 0; start < points.size();) {
            int cindex = (start + (span / 2)) % points.size();
            int eindex = (start + span) % points.size();
            
            cv::Point s = points[start];
            cv::Point c = points[cindex];
            cv::Point e = points[eindex];
            
            float angle = calcThreePointAngle(s, c, e);
            if (0 <= angle && angle < 75) {
                cv::line(rgb, s, c, cv::Scalar(0, 0, 255), 2);
                cv::line(rgb, c, e, cv::Scalar(0, 0, 255), 2);
                cv::circle(rgb, c, 5, cv::Scalar(255, 255, 0), -1);
                start += (span / 2);
            } else {
                start++;
            }
        }
    }
}

int main(int argc, const char * argv[]) {
    // カメラを開く
    auto cap = cv::VideoCapture(0);
    if (!cap.isOpened()) {
        fprintf(stderr, "Failed to open camera\n");
        return -1;
    }
    
    // 画像処理部
    while (true) {
        cv::Mat input, output;
        
        // カメラから画像を取得
        cap >> input;
        
        // 画像処理を実施する
        doProcess(input, output);
        
        // 処理結果の表示
        cv::imshow("frame", output);
        int key = cv::waitKey(1);
        if (key == 27) {
            break;
        }
    }
    return 0;
}
