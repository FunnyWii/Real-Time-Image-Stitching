// Based on OpenCV 4.5.4
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif
#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

// Default command line args
// TODO: Using config file to read
std::vector<cv::String> img_names;
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 0.5f;
#ifdef HAVE_OPENCV_XFEATURES2D
std::string features_type = "surf";
float match_conf = 0.65f;
#else
std::string features_type = "orb";
float match_conf = 0.3f;
#endif
std::string matcher_type = "homography";
std::string estimator_type = "homography";
std::string ba_cost_func = "ray";
std::string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
cv::detail::WaveCorrectKind wave_correct = cv::detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
std::string warp_type = "spherical";
int expos_comp_type = cv::detail::ExposureCompensator::GAIN_BLOCKS;
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 32;
std::string seam_find_type = "gc_color";
int blend_type = cv::detail::Blender::MULTI_BAND;
int timelapse_type = cv::detail::Timelapser::AS_IS;
float blend_strength = 5;
std::string result_name = "../res/result.jpg";
bool timelapse = false;
int range_width = -1;


int main (int argc, char* argv[])
{
    // Assume we know the position of the camera
    img_names.push_back("../imgs/l70.jpg");
    img_names.push_back("../imgs/lr0.jpg");
    img_names.push_back("../imgs/r70.jpg");
    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        std::cerr << "Need at least two images to stitch" << std::endl;
        return -1;
    }

    // Set default scales
    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
    
    // ------------------- Find features ----------------------
    LOGLN("Finding features...");
#if ENABLE_LOG
    int64 t = cv::getTickCount();
#endif
    cv::Ptr<cv::Feature2D> finder;
    // TODO: Model selection
    finder = cv::AKAZE::create();

    cv::Mat full_img, img;
    std::vector<cv::detail::ImageFeatures> features(num_images);
    std::vector<cv::Mat> images(num_images);
    std::vector<cv::Size> full_img_sizes(num_images);
    double seam_work_aspect = 1.0;
    for (int i = 0; i < num_images; ++i){
        full_img = cv::imread(cv::samples::findFile(img_names[i]));
        full_img_sizes[i] = full_img.size();
        if (full_img.empty()){
            LOGLN("Can't open image " << img_names[i]);
            return -1;
        }
        if (work_megapix < 0){
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else{
            if (!is_work_scale_set){
                work_scale = std::min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            cv::resize(full_img, img, cv::Size(), work_scale, work_scale, cv::INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set){
            seam_scale = std::min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }
        cv::detail::computeImageFeatures(finder, img, features[i]);    
        features[i].img_idx = i;
        LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());
        cv::resize(full_img, img, cv::Size(), seam_scale, seam_scale, cv::INTER_LINEAR_EXACT); 
        images[i] = img.clone();    
    }
    full_img.release();
    img.release();
    LOGLN("Finding features, time: " << ((cv::getTickCount() - t) /cv::getTickFrequency()) << " sec");


    // ------------------- Matching ---------------------
    LOG("Pairwise matching");
#if ENABLE_LOG
    t = cv::getTickCount();
#endif
    std::vector<cv::detail::MatchesInfo> pairwise_matches;
    cv::Ptr<cv::detail::FeaturesMatcher> matcher; 
    // TODO: Model selection
    matcher = cv::makePtr<cv::detail::BestOf2NearestMatcher>(try_cuda, match_conf);
    (*matcher)(features, pairwise_matches); 
    matcher->collectGarbage();
    LOGLN("Pairwise matching, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");
    if (save_graph){
        LOGLN("Saving matches graph...");
        std::ofstream f(save_graph_to.c_str());
        f << cv::detail::matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
    }
    std::vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh); // 保存的是图像拼接的顺序
    std::vector<cv::Mat> img_subset; 
    std::vector<cv::String> img_names_subset;
    std::vector<cv::Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i){
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }
    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;
    num_images = static_cast<int>(img_names.size());
    if (num_images < 2){
        LOGLN("After Pair-wise matching, only one image is left");
        return -1;
    }

    cv::Ptr<cv::detail::Estimator> estimator;       
    // TODO: Model selection
    estimator = cv::makePtr<cv::detail::HomographyBasedEstimator>();
    std::vector<cv::detail::CameraParams> cameras;
    if (!(*estimator)(features, pairwise_matches, cameras)){
        std::cout << "Homography estimation failed.\n";
        return -1;
    }
    for (size_t i = 0; i < cameras.size(); ++i){
        cv::Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        LOGLN("Initial camera intrinsics #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
    }
    cv::Ptr<cv::detail::BundleAdjusterBase> adjuster;
    // TODO: Model selection
    adjuster = cv::makePtr<cv::detail::BundleAdjusterRay>();
    adjuster->setConfThresh(conf_thresh);
    cv::Mat_<uchar> refine_mask = cv::Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
    adjuster->setRefinementMask(refine_mask);
    if (!(*adjuster)(features, pairwise_matches, cameras)){
        std::cout << "Camera parameters adjusting failed.\n";
        return -1;
    }

    std::vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i){
        LOGLN("Camera #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
        focals.push_back(cameras[i].focal);
    }
    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
    std::cout << "warped_image_scale: " << warped_image_scale << std::endl;
    if (do_wave_correct){
        std::vector<cv::Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }

    // -------------------- warp -------------------------
    LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
    t = cv::getTickCount();
#endif
    std::vector<cv::Point> corners(num_images);
    std::vector<cv::UMat> masks_warped(num_images);
    std::vector<cv::UMat> images_warped(num_images);
    std::vector<cv::Size> sizes(num_images);
    std::vector<cv::UMat> masks(num_images);
    for (int i = 0; i < num_images; ++i){
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(cv::Scalar::all(255));
    }
    cv::Ptr<cv::WarperCreator> warper_creator;
    // TODO: Model selection
    warper_creator = cv::makePtr<cv::SphericalWarper>();
    if (!warper_creator){
        std::cout << "Can't create the following warper '" << warp_type << "'\n";
        return 1;
    }
    cv::Ptr<cv::detail::RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
    for (int i = 0; i < num_images; ++i){
        cv::Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;
        corners[i] = warper->warp(images[i], K, cameras[i].R,  cv::INTER_LINEAR,  cv::BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();
        warper->warp(masks[i], K, cameras[i].R,  cv::INTER_NEAREST,  cv::BORDER_CONSTANT, masks_warped[i]);
       
    }
    std::vector<cv::UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);
    LOGLN("Warping images, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");
    LOGLN("Compensating exposure...");
#if ENABLE_LOG
    t = cv::getTickCount();
#endif
    cv::Ptr<cv::detail::ExposureCompensator> compensator = cv::detail::ExposureCompensator::createDefault(expos_comp_type);
    if (dynamic_cast<cv::detail::GainCompensator*>(compensator.get()))
    {
        cv::detail::GainCompensator* gcompensator = dynamic_cast<cv::detail::GainCompensator*>(compensator.get());
        gcompensator->setNrFeeds(expos_comp_nr_feeds);
    }
    if (dynamic_cast<cv::detail::ChannelsCompensator*>(compensator.get()))
    {
        cv::detail::ChannelsCompensator* ccompensator = dynamic_cast<cv::detail::ChannelsCompensator*>(compensator.get());
        ccompensator->setNrFeeds(expos_comp_nr_feeds);
    }
    if (dynamic_cast<cv::detail::BlocksCompensator*>(compensator.get()))
    {
        cv::detail::BlocksCompensator* bcompensator = dynamic_cast<cv::detail::BlocksCompensator*>(compensator.get());
        bcompensator->setNrFeeds(expos_comp_nr_feeds);
        bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
        bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
    }
    compensator->feed(corners, images_warped, masks_warped);
    LOGLN("Compensating exposure, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");

    // ----------------- finding seams -------------------------------
    LOGLN("Finding seams...");
#if ENABLE_LOG
    t = cv::getTickCount();
#endif
    cv::Ptr<cv::detail::SeamFinder> seam_finder;        
    // TODO: Model selection
    seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
    if (!seam_finder){
        std::cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
        return 1;
    }
    seam_finder->find(images_warped_f, corners, masks_warped);
    LOGLN("Finding seams, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");
    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    // ----------------- Compositing -------------------------------
    LOGLN("Compositing...");

    // Loop
    int loop_count = 0;
    cv::Mat img_warped, img_warped_s;
    cv::Mat dilated_mask, seam_mask, mask, mask_warped;
    cv::Ptr<cv::detail::Blender> blender;
    cv::Ptr<cv::detail::Timelapser> timelapser;
    double compose_work_aspect = 1;
    if (!is_compose_scale_set)
    {
        if (compose_megapix > 0) // 暂时不可以设置compose_megapix，full_img.size()使用的之前已经release的，而不是下面重新读取的；目前不会运行到这行
            compose_scale = std::min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
        is_compose_scale_set = true;
        // Compute relative scales
        compose_work_aspect = compose_scale / work_scale;
        // Update warped image scale
        warped_image_scale *= static_cast<float>(compose_work_aspect);
        warper = warper_creator->create(warped_image_scale);
        // Update corners and sizes
        for (int i = 0; i < num_images; ++i)
        {
            // Update intrinsics
            cameras[i].focal *= compose_work_aspect;
            cameras[i].ppx *= compose_work_aspect;
            cameras[i].ppy *= compose_work_aspect;
            // Update corner and size
            cv::Size sz = full_img_sizes[i];
            if (std::abs(compose_scale - 1) > 1e-1)
            {
                sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                sz.height = cvRound(full_img_sizes[i].height * compose_scale);
            }
            cv::Mat K;
            cameras[i].K().convertTo(K, CV_32F);
            cv::Rect roi = warper->warpRoi(sz, K, cameras[i].R);
            corners[i] = roi.tl();
            sizes[i] = roi.size();
        }
    }
    std::vector<cv::Mat> intrinsicMat(num_images);
    std::vector<cv::Mat> temp_dilated_mask(num_images);
    std::vector<cv::Mat> temp_mask_warped(num_images);
    std::vector<cv::Mat> temp_seam_mask(num_images);
    std::vector<cv::Mat> saved_final_mask_warped(num_images);
    cv::Size origin_size = cv::Size(1920,1080);
    if (abs(compose_scale - 1) > 1e-1){
        origin_size.width = static_cast<int>(origin_size.width * compose_scale);
        origin_size.height = static_cast<int>(origin_size.height * compose_scale);
    }
    for(int i = 0; i < num_images; ++i){
        cameras[i].K().convertTo(intrinsicMat[i], CV_32F);
        cv::Mat temp_mask;
        temp_mask.create(origin_size, CV_8U);
        temp_mask.setTo(cv::Scalar::all(255));
        warper->warp(temp_mask, intrinsicMat[i], cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, temp_mask_warped[i]); 
        dilate(masks_warped[i], temp_dilated_mask[i], cv::Mat());
        cv::resize(temp_dilated_mask[i], temp_seam_mask[i], temp_mask_warped[i].size(), 0, 0, cv::INTER_LINEAR_EXACT);
        saved_final_mask_warped[i] = temp_seam_mask[i] & temp_mask_warped[i];
        // cv::imwrite("../res/loop" + std::to_string(loop_count) + "saved_final_mask_warped" + std::to_string(i) + ".jpg", saved_final_mask_warped[i]);
        // TODO : release 
    }

    if (!blender && !timelapse) {
        blender = cv::detail::Blender::createDefault(blend_type, try_cuda);
        cv::Size dst_sz = cv::detail::resultRoi(corners, sizes).size();
        float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
        if (blend_width < 1.f)
            blender = cv::detail::Blender::createDefault(cv::detail::Blender::NO, try_cuda);
        else if (blend_type == cv::detail::Blender::MULTI_BAND)
        {
            cv::detail::MultiBandBlender* mb = dynamic_cast<cv::detail::MultiBandBlender*>(blender.get());
            mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
            LOGLN("Multi-band blender, number of bands: " << mb->numBands());
        }
        else if (blend_type == cv::detail::Blender::FEATHER)
        {
            cv::detail::FeatherBlender* fb = dynamic_cast<cv::detail::FeatherBlender*>(blender.get());
            fb->setSharpness(1.f/blend_width);
            LOGLN("Feather blender, sharpness: " << fb->sharpness());
        }
        // blender->prepare(corners, sizes);
    }
    else if (!timelapser && timelapse){
        timelapser = cv::detail::Timelapser::createDefault(timelapse_type);
        timelapser->initialize(corners, sizes);
    }
    // ------------------------ INTO WHILE LOOP
    while(true){
    #if ENABLE_LOG
        t = cv::getTickCount();
    #endif
        auto start_time = std::chrono::high_resolution_clock::now();
        blender->prepare(corners, sizes); // 12ms, 因为每次blender都release了内部的资源，所以需要重新prepare，看看能否不释放？
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << " blender->prepare Elapsed time: " << duration << " ms" << std::endl;
        for (int img_idx = 0; img_idx < num_images; ++img_idx){
            LOGLN("Compositing image #" << indices[img_idx]+1);
            full_img = cv::imread(cv::samples::findFile(img_names[img_idx]));   

            if (abs(compose_scale - 1) > 1e-1)
                cv::resize(full_img, img, cv::Size(), compose_scale, compose_scale, cv::INTER_LINEAR_EXACT);
            else
                img = full_img;
            std::cout << "full_img size: " << full_img.size() << std::endl;
            std::cout << "img size: " << img.size() << std::endl;
            // cv::imwrite("../res/loop" + std::to_string(loop_count) + "img" + std::to_string(img_idx) + ".jpg", img);
            full_img.release();
            // LOGLN("full img relase, img get ...");
            cv::Size img_size = img.size();
            // cv::Mat K;
            // cameras[img_idx].K().convertTo(K, CV_32F);
            // Warp the current image
            start_time = std::chrono::high_resolution_clock::now();
            warper->warp(img, intrinsicMat[img_idx], cameras[img_idx].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, img_warped);
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            std::cout << "IMG  warper->warp Elapsed time: " << duration << " ms" << std::endl;
            // cv::imwrite("../res/loop" + std::to_string(loop_count) + "img_warped" + std::to_string(img_idx) + ".jpg", img_warped);
            // LOGLN("img warped");
            // Warp the current image mask
            // mask.create(img_size, CV_8U);
            // mask.setTo(cv::Scalar::all(255));
            // start_time = std::chrono::high_resolution_clock::now();
            // warper->warp(mask, intrinsicMat[img_idx], cameras[img_idx].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, mask_warped); // TODO：mask不需要实时计算
            // cv::imwrite("../res/loop" + std::to_string(loop_count) + "mask_warped" + std::to_string(img_idx) + ".jpg", mask_warped);
            // LOGLN("mask warped");
            // end_time = std::chrono::high_resolution_clock::now();
            // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            // std::cout << " MASK warper->warp Elapsed time: " << duration << " ms" << std::endl;
            // Compensate exposure
            compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
            img_warped.convertTo(img_warped_s, CV_16S);
            img_warped.release();
            img.release();
            // mask.release();
            // LOGLN("resource released....");
            // start_time = std::chrono::high_resolution_clock::now();
            // dilate(masks_warped[img_idx], dilated_mask, cv::Mat());
            // end_time = std::chrono::high_resolution_clock::now();
            // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            // std::cout << " dilate Elapsed time: " << duration << " ms" << std::endl;
            // cv::imwrite("../res/loop" + std::to_string(loop_count) + "dilated_mask" + std::to_string(img_idx) + ".jpg", dilated_mask);
            // cv::resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, cv::INTER_LINEAR_EXACT);
            // cv::imwrite("../res/loop" + std::to_string(loop_count) + "seam_mask" + std::to_string(img_idx) + ".jpg", seam_mask);
            // mask_warped = seam_mask & mask_warped;
            // cv::imwrite("../res/loop" + std::to_string(loop_count) + "mask_warped_seam_mask" + std::to_string(img_idx) + ".jpg", mask_warped);
            // LOGLN("final mask generated....");

            
            // LOGLN("NO timelapse, blender feed");
            // LOGLN("--------------corners[" << img_idx << "]: " << corners[img_idx] << "corner size" << corners.size()); // 打印 corners[img_idx]
            // LOGLN("--------------size [" << img_idx << "]: " << sizes[img_idx] << "sizes size" << sizes.size()); // 打印 sizes[img_idx]
            // blender->prepare(corners, sizes);
            start_time = std::chrono::high_resolution_clock::now();
            blender->feed(img_warped_s, saved_final_mask_warped[img_idx], corners[img_idx]); // 未应用blender,结果保存在blender->blend(result, result_mask)中
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            std::cout << " blender->feed Elapsed time: " << duration << " ms" << std::endl;

            // LOGLN("blender feed done");
        }
        if (!timelapse){
            cv::Mat result, result_mask;
            start_time = std::chrono::high_resolution_clock::now();
            blender->blend(result, result_mask);
            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            std::cout << " blender->blend Elapsed time: " << duration << " ms" << std::endl;
            // img_warped_s.release();
            // seam_mask.release();
            LOGLN("Compositing, time: " << ((cv::getTickCount() - t) / cv::getTickFrequency()) << " sec");
            LOGLN("----------------------------------------------------------------------------------");
            cv::imwrite("../res/res" + std::to_string(loop_count) + ".jpg",result);
            loop_count ++;
            // cv::imshow("result", result);
            // TODO : crop image, manual selection
        }
        if (cv::waitKey(1) == 27)
            break;
    }


    return 0;
}