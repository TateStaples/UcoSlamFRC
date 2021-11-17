#include <iostream>
#include <math.h>
#include "ucoslam.h"
#include "mapviewer.h"

class CmdLineParser {
    int argc; char** argv;
    public: CmdLineParser(int _argc, char** _argv) :argc(_argc), argv(_argv) {}  bool operator[] (string param) { int idx = -1;  for (int i = 0; i < argc && idx == -1; i++) if (string(argv[i]) == param) idx = i;    return (idx != -1); } string operator()(string param, string defvalue = "") { int idx = -1;    for (int i = 0; i < argc && idx == -1; i++) if (string(argv[i]) == param) idx = i; if (idx == -1) return defvalue;   else  return (argv[idx + 1]); }
      std::vector<std::string> getAllInstances(string str) {
          std::vector<std::string> ret;
          for (int i = 0; i < argc - 1; i++) {
              if (string(argv[i]) == str)
                  ret.push_back(argv[i + 1]);
          }
          return ret;
      }
};

void overwriteParamsByCommandLine(CmdLineParser& cml, ucoslam::Params& params) {
    if (cml["-aruco-markerSize"])      params.aruco_markerSize = stof(cml("-aruco-markerSize", "1"));
    if (cml["-marker_minsize"])    params.aruco_minMarkerSize = stod(cml("-marker_minsize", "0.025"));
    if (cml["-nokeypoints"])params.detectKeyPoints = false;
    if (cml["-nomarkers"])  params.detectMarkers = false;
    if (cml["-sequential"]) params.runSequential = true;
    if (cml["s"])    params.maxFeatures = stoi(cml("-maxFeatures", "4000"));
    if (cml["-nOct"])       params.nOctaveLevels = stoi(cml("-nOct", "8"));
    if (cml["-fdt"])        params.nthreads_feature_detector = stoi(cml("-fdt", "2"));
    if (cml["-desc"])       params.kpDescriptorType = ucoslam::DescriptorTypes::fromString(cml("-desc", "orb"));
    if (cml["-dict"])       params.aruco_Dictionary = cml("-dict");
    if (cml["-tfocus"])  params.targetFocus = stof(cml("-tfocus", "-1"));
    if (cml["-KFMinConfidence"])  params.KFMinConfidence = stof(cml("-KFMinConfidence"));
    if (cml["s"])    params.KPNonMaximaSuppresion = true;

    if (cml["-autoAdjustKpSensitivity"])    params.autoAdjustKpSensitivity = true;
    if (cml["-extra_params"])    params.extraParams = cml("-extra_params");

    if (cml["-scale"]) params.kptImageScaleFactor = stof(cml("-scale"));

    if (cml["-nokploopclosure"]) params.reLocalizationWithKeyPoints = false;
    if (cml["-inplanemarkers"]) params.inPlaneMarkers = true;
    params.aruco_CornerRefimentMethod = cml("-aruco-cornerRefinementM", "CORNER_SUBPIX");
}

int main(int argc, char** argv) {
    try {
        CmdLineParser cml(argc, argv);
        ucoslam::UcoSlam Slam;
        int debugLevel = stoi(cml("-debug", "0"));
        Slam.setDebugLevel(debugLevel);
        Slam.showTimers(true);
        ucoslam::ImageParams image_params;
        ucoslam::Params params;
        cv::Mat in_image;


        image_params.readFromXMLFile(cml("-camConfigPath", "./data/limelight.yml"));

        params.readFromYMLFile(cml("-params", "./data/fileout.yml"));
        overwriteParamsByCommandLine(cml, params);

        auto TheMap = std::make_shared<ucoslam::Map>();
        //read the map from file?
        if (cml["-map"]) TheMap->readFromFile(cml("-map"));

        Slam.setParams(TheMap, params, cml("-voc", "./data/orb.fbow"));


        if (cml["-loc_only"]) Slam.setMode(ucoslam::MODE_LOCALIZATION);

        //Ok, lets start
        int currentFrameIndex = 0;
        cv::Mat camPose_c2g;
        string storagePath = cml("-valJson", "slamValues.json");
        double newUpdate; double prevUpdate = -1.0;
        bool running = true;

        while (running) {
            cv::FileStorage reader(storagePath, cv::FileStorage::READ);
            reader["newImgTime"] >> newUpdate;
            reader["running"] >> running;
            cerr << newUpdate << endl;
            cerr << "loop" << endl;
            if (newUpdate == prevUpdate) continue;  // check if the value has been updated
            prevUpdate = newUpdate;
            in_image = cv::imread("slamImage.jpg");
            if (in_image.empty()) continue;
            camPose_c2g = Slam.process(in_image, image_params, currentFrameIndex);
            cerr << camPose_c2g << endl;
            if (!camPose_c2g.empty()) {
                reader["newImgTime"] >> newUpdate;
                reader["running"] >> running;
                reader.release();
                cv::FileStorage writer(storagePath, cv::FileStorage::WRITE);
                double x = camPose_c2g.at<double>(0, 3);
                double y = camPose_c2g.at<double>(2, 3);
                double theta = atan2(camPose_c2g.at<double>(2, 1), camPose_c2g.at<double>(2, 2));
                writer.write("X", x);
                writer.write("Y", y);
                writer.write("THETA", theta);
                writer.write("outputImgTime", prevUpdate);
                writer.write("running", false);
                writer.write("newImageTime", newUpdate);
                writer.release();
            }
            else reader.release();
            currentFrameIndex++;
        }

        //save the output

        TheMap->saveToFile(cml("-out", "world") + ".map");
    }
    catch (std::exception& ex) {
        cerr << ex.what() << endl;
    }
}