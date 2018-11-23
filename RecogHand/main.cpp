//
//  main.cpp
//  RecogHand
//
//  Created by 中山一輝 on 2018/11/19.
//  Copyright © 2018年 中山一輝. All rights reserved.
//

// INCLUDE
#include "Reader.h"
#include "Processor.h"

int main(int argc, const char * argv[]) {
    IReader* reader = NULL;
    if (argc == 2) {
        reader = new SingleImageReader(argv[1]);
    } else {
        reader = new VideoReader(0);
    }

    if (reader == NULL || !reader->isOpened()) {
        fprintf(stderr, "Failed to open camera\n");
        delete reader;
        return -1;
    }
    
    // 画像処理部
    while (true) {
        cv::Mat input, output;
        
        // カメラから画像を取得
        *reader >> input;
        if (input.data == NULL) {
            fprintf(stderr, "Error in reader\n");
            break;
        }
        
        // 画像処理を実施する
        Processor::recog(input, output);
        
        // 処理結果の表示
        cv::imshow("frame", output);
        int key = cv::waitKey(1);
        if (key == 27) {
            break;
        } else if (key == 's') {
            std::string filename = std::to_string(time(NULL)) + ".png";
            cv::imwrite("/Users/nakayama/" + filename, input);
        }
    }
    
    delete reader;
    
    return 0;
}
