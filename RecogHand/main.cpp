//
//  main.cpp
//  RecogHand
//
//  Created by 中山一輝 on 2018/11/19.
//  Copyright © 2018年 中山一輝. All rights reserved.
//

#include <opencv2/opencv.hpp>


void extractSkin(const cv::Mat& frame, cv::Mat& dst) {
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
    
    for (int i = 0; i < contours.size(); i++){
        cv::drawContours(rgb, contours, i, CV_RGB(0, 0, 255), 4);
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
