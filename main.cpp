//
//  main.cpp
//  depth_fusion
//
//  Created by Yusuke Yasui on 4/23/18.
//  Copyright © 2018 Yusuke Yasui. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

#include "omp.h"

//#define DEBUG

using namespace std;
using namespace cv;

static float toRadian(const float degree)
{
    return degree / 180.0 * M_PI;
}

static cv::Mat ConvertToHeatmap(const cv::Mat &src, ushort maxVal)
{
    cv::Mat srcTrueDepth8U;
    src.convertTo(srcTrueDepth8U, CV_8U, 1.0/maxVal * 255.0);
        
    cv::Mat heatmap;
    applyColorMap(srcTrueDepth8U, heatmap, COLORMAP_HSV);
    
    return heatmap;
}

// Compute barycentric coordinates (u, v, w) for point p with respect to triangle (a, b, c)
// Copied from https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
static void Barycentric(Point2f p, Point2f a, Point2f b, Point2f c, float &u, float &v, float &w)
{
    Point2f v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = v0.dot(v0);
    float d01 = v0.dot(v1);
    float d11 = v1.dot(v1);
    float d20 = v2.dot(v0);
    float d21 = v2.dot(v1);
    float denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;
}

class DepthEdgeAlignment
{
    cv::Mat srcCamera;
    cv::Mat cameraToImage;
    std::vector<cv::Point3f> lidarPoints;
    cv::Mat edgeWeightX, edgeWeightY;
    cv::Mat smoothingWeight;
    
    cv::Mat InterpolateLidarPoints(const cv::Mat &srcLidar,
                                   std::vector<ushort> &depthFromLidar,
                                   std::vector<int>    &lidarPixelIndex,
                                   cv::Mat             &reliableRegion,
                                   int reliableRadius = 5);
    
    cv::Mat PrimalDualHybridGradient(const cv::Mat &src, double lambda,
                                     const std::vector<ushort> &depthFromLidar,
                                     const std::vector<int>    &lidarPixelIndex,
                                     const cv::Mat &smoothingWeight,
                                     int maxIter = 1000);
    
    void ProjectOntoUnitBall(cv::Mat &x);
    
    cv::Mat Aty(const cv::Mat &y, int nRows, int nCols, const cv::Mat &smoothingWeight = cv::Mat());
    cv::Mat Ax(const cv::Mat &x, int nRows, int nCols, const cv::Mat &smoothingWeight = cv::Mat());
    
    void GetSmoothingWeights(double tau, double gamma);
    
    double EstimateCalibrationFidelity(const cv::Mat &estimatedDepth, const cv::Mat &reliableRegion);
    
public:
    void   ReadData(const std::string &cameraPath,
                    const std::string &lidarPath,
                    const std::string &calibrationPath,
                    const std::string &dataId,
                    cv::Mat &lidarToCamera);
    cv::Mat FindCalibrationParameters(const cv::Vec6f &initCalibrationParams, const cv::Vec6f &neighborDelta);
    double  DepthFusion(const cv::Mat &lidarToCamera, bool isAlignEdge, cv::Mat &optimizedDepth, int nIter = 0);
};


void DepthEdgeAlignment::ReadData(const std::string &cameraPath,
                                  const std::string &lidarPath,
                                  const std::string &calibrationPath,
                                  const std::string &dataId,
                                  cv::Mat &lidarToCamera)
{
    ifstream fin;
    
    // read point clouds from lidar
    fin.open(lidarPath + dataId + ".bin", ios::binary);
    if (fin)
    {
        fin.seekg (0, fin.end);
        int fileSize = static_cast<int>(fin.tellg());
        fin.seekg (0, fin.beg);
        
        cerr << "file size " << fileSize << " bytes." << endl;
        
        CV_Assert(fileSize%16 == 0);
        int nPoints = fileSize / 16; // each point consists of 4 floats (= 4 bytes)
        
        for(int i = 0; i < nPoints; ++i)
        {
            float x, y, z, reflectance;
            
            fin.read((char*)&x, sizeof(float));
            fin.read((char*)&y, sizeof(float));
            fin.read((char*)&z, sizeof(float));
            fin.read((char*)&reflectance, sizeof(float));
            
            lidarPoints.emplace_back(cv::Point3f(x, y, z));
        }
    }
    
    fin.close();
    
    // read calibration data
    // For the file format, refer to http://www.mrt.kit.edu/z/publ/download/2013/GeigerAl2013IJRR.pdf
    
    cv::Mat projectionMat(3, 4, CV_32F), RectifyMat = cv::Mat::zeros(4, 4, CV_32F);
    
    lidarToCamera = cv::Mat::zeros(4, 4, CV_32F);
    
    fin.open(calibrationPath + dataId + ".txt");
    if (fin)
    {
        // extract camera index from cameraPath
        CV_Assert(cameraPath.find_last_of("_") != string::npos);
        std::string cameraId = "P" + cameraPath.substr(cameraPath.find_last_of("_") + 1, 1);
        
        cerr << "cameraId " << cameraId << endl;
        
        auto fillMatrixData = [](cv::Mat m, const std::vector<std::string> &tokens)
        {
            int index = 1;
            for(int y = 0; y < m.rows; ++y)
            {
                for(int x = 0; x < m.cols; ++x)
                {
                    m.at<float>(y, x) = stof(tokens[index++]);
                }
            }
        };
        
        std::string inputLine;
        
        while(std::getline(fin, inputLine))
        {
            std::vector<std::string> tokens;
            size_t curPos = 0, endPos;
            
            while ((endPos = inputLine.find(" ", curPos)) != std::string::npos)
            {
                tokens.emplace_back(inputLine.substr(curPos, endPos - curPos));
                curPos = endPos + 1;
            }
            
            tokens.emplace_back(inputLine.substr(curPos));
            
            if(!tokens.empty())
            {
                if(!tokens[0].compare(cameraId + ":"))
                {
                    fillMatrixData(projectionMat, tokens);
                }
                else if(!tokens[0].compare("R0_rect:"))
                {
                    fillMatrixData(RectifyMat(cv::Rect(0, 0, 3, 3)), tokens);
                    RectifyMat.at<float>(3, 3) = 1.0;
                }
                else if(!tokens[0].compare("Tr_velo_to_cam:"))
                {
                    fillMatrixData(lidarToCamera(cv::Rect(0, 0, 4, 3)), tokens);
                    lidarToCamera.at<float>(3, 3) = 1.0;
                }
            }
        }
    }
    
    fin.close();
    
    cerr << projectionMat << endl;
    cerr << RectifyMat << endl;
    cerr << lidarToCamera << endl;
    
    cameraToImage = projectionMat * RectifyMat;
    
    srcCamera = imread(cameraPath + dataId + ".png", CV_LOAD_IMAGE_ANYDEPTH);
    
    // small tau -> smoother. higher tau -> more sensitive to small edges
    GetSmoothingWeights(0.1, 2.0);
    
#ifdef DEBUG
    imwrite("srcCamera.png", srcCamera);
#endif
}



// Optimization based on "An Efficient Primal-Dual Hybrid Gradient Algorithm For Total Variation Image Restoration"
// By Mingqiang Zhu and Tony Chan
cv::Mat DepthEdgeAlignment::PrimalDualHybridGradient(const cv::Mat &src, double lambda,
                                                     const std::vector<ushort> &depthFromLidar,
                                                     const std::vector<int>    &lidarPixelIndex,
                                                     const cv::Mat &smoothingWeight,
                                                     int maxIter)
{
    // invert lambda since in the original paper, lambda is set to regularize term while,
    // in Primal-Dual Hybrid Gradient paper, lambad is set to "faithful to input image" term
    lambda = 1.0 / lambda;
    
    int nRows = src.rows;
    int nCols = src.cols;
    
    cv::Mat y; // optimized output
    
    src.reshape(0, src.cols * src.rows).convertTo(y, CV_64F);
    
    CV_Assert(depthFromLidar.size() == lidarPixelIndex.size());
    
    cv::Mat x = cv::Mat::zeros(y.rows*2, 1, CV_64F);
    
    cv::Mat Btz = cv::Mat::zeros(y.size(), y.type());
    
    for(size_t i = 0; i < lidarPixelIndex.size(); ++i)
    {
        Btz.at<double>(lidarPixelIndex[i]) = static_cast<double>(depthFromLidar[i]);
    }
    
    for(int k = 0; k <= maxIter; ++k) // add stop criterion based on paper later..
    {
        double tau   = 0.2 + 0.08*k;
        double theta = 0.5 / tau;
        
        cv::Mat x1 = x + tau * Aty(y, nRows, nCols, smoothingWeight);
        
        ProjectOntoUnitBall(x1);
        
        cv::Mat y1 = y - theta * (Ax(x1, nRows, nCols, smoothingWeight) - lambda*Btz);
        
        double invOnePlusThetaLambda = 1.0 / (1.0 + theta * lambda);
        
        for(size_t i = 0; i < lidarPixelIndex.size(); ++i)
        {
            y1.at<double>(lidarPixelIndex[i]) *= invOnePlusThetaLambda;
        }
        
        y = y1;
        x = x1;
    }
    
    cv::Mat optimized;
    
    y.reshape(0, src.rows).convertTo(optimized, src.type());
    
    return optimized;
}


void DepthEdgeAlignment::ProjectOntoUnitBall(cv::Mat &x)
{
#pragma omp parallel for
    for(int i = 0; i < x.rows; i += 2)
    {
        double &xx = x.at<double>(i);
        double &xy = x.at<double>(i + 1);
        
        double norm = sqrt(xx*xx + xy*xy);

        if(norm > 1.0)
        {
            double invNorm = 1.0 / norm;
            
            xx *= invNorm;
            xy *= invNorm;
        }
    }
}

cv::Mat DepthEdgeAlignment::Aty(const cv::Mat &y, int nRows, int nCols, const cv::Mat &smoothingWeight)
{
    cv::Mat Aty = Mat::zeros(nRows*nCols*2, 1, CV_64F);
    
    // Not the last row/column
#pragma omp parallel for
    for(int j = 0; j < nRows - 1; ++j)
    {
        int index = j*nCols;
        for(int i = 0; i < nCols - 1; ++i)
        {
            double weight = (!smoothingWeight.empty() ? smoothingWeight.at<double>(index) : 1.0);
            
            Aty.at<double>(index*2)   = (y.at<double>(index+1)     - y.at<double>(index)) * weight;
            Aty.at<double>(index*2+1) = (y.at<double>(index+nCols) - y.at<double>(index)) * weight;
            
            ++index;
        }
    }
    
    // last row
    {
        int index = (nRows-1)*nCols;
        for(int i = 0; i < nCols - 1; ++i)
        {
            double weight = (!smoothingWeight.empty() ? smoothingWeight.at<double>(index) : 1.0);
            
            Aty.at<double>(index*2)   = (y.at<double>(index+1) - y.at<double>(index)) * weight;
            ++index;
        }
    }
    
    // last column
    {
        int index = nCols - 1;
        for(int j = 0; j < nRows - 1; ++j)
        {
            double weight = (!smoothingWeight.empty() ? smoothingWeight.at<double>(index) : 1.0);
            
            Aty.at<double>(index*2+1) = (y.at<double>(index+nCols) - y.at<double>(index)) * weight;
            index += nCols;
        }
    }
    
    return Aty;
}

cv::Mat DepthEdgeAlignment::Ax(const cv::Mat &x, int nRows, int nCols, const cv::Mat &smoothingWeight)
{
    cv::Mat Ax = Mat::zeros(nRows*nCols, 1, CV_64F);

    // Not the last row/column
#pragma omp parallel for
    for(int j = 0; j < nRows - 1; ++j)
    {
        int index = j*nCols;
        for(int i = 0; i < nCols - 1; ++i)
        {
            double weight = (!smoothingWeight.empty() ? smoothingWeight.at<double>(index) : 1.0);
            
            Ax.at<double>(index)       += (-x.at<double>(index*2) - x.at<double>(index*2 + 1)) * weight;
            Ax.at<double>(index+1)     += x.at<double>(index*2)     * weight;
            Ax.at<double>(index+nCols) += x.at<double>(index*2 + 1) * weight;
            
            ++index;
        }
    }

    // for last row
    {
        int index = (nRows-1)*nCols;
        for(int i = 0; i < nCols - 1; ++i)
        {
            double weight = (!smoothingWeight.empty() ? smoothingWeight.at<double>(index) : 1.0);
            
            Ax.at<double>(index)   += (-x.at<double>(index*2)) * weight;
            Ax.at<double>(index+1) += x.at<double>(index*2)    * weight;
            ++index;
        }
    }

    // for last column
    {
        int index = nCols - 1;
        for(int j = 0; j < nRows - 1; ++j)
        {
            double weight = (!smoothingWeight.empty() ? smoothingWeight.at<double>(index) : 1.0);
            
            Ax.at<double>(index)       += (x.at<double>(index*2 + 1)) * weight;
            Ax.at<double>(index+nCols) += x.at<double>(index*2 + 1)   * weight;
            index += nCols;
        }
    }

    return Ax;
}


double DepthEdgeAlignment::EstimateCalibrationFidelity(const cv::Mat &estimatedDepth, const cv::Mat &reliableRegion)
{
    double Ax = 0.0, Ay = 0.0;
    double edgeWeightSumX = 0.0, edgeWeightSumY = 0.0;
    double depthGradSumX = 0.0, depthGradSumY = 0.0;
    
    for(int y = 0; y < estimatedDepth.rows - 1; ++y)
    {
        const double*  d1 = estimatedDepth.ptr<double>(y);
        const double*  d2 = estimatedDepth.ptr<double>(y + 1);
        const double* ewx = edgeWeightX.ptr<double>(y);
        const double* ewy = edgeWeightY.ptr<double>(y);
        const uchar*    r = reliableRegion.ptr<uchar>(y);
        
        for(int x = 0; x < estimatedDepth.cols - 1; ++x)
        {
            if(*r++ > 0)
            {
                // only consider region surronded by lidar data
                double gradX = fabs(*(d1 + 1) - *d1);
                double gradY = fabs(*d2 - *d1);
                
                Ax             += (*ewx * gradX);
                edgeWeightSumX += *ewx;
                depthGradSumX  += gradX;

                Ay             += (*ewy * gradY);
                edgeWeightSumY += *ewy;
                depthGradSumY  += gradY;
            }
            
            ++d1;
            ++d2;
            ++ewx;
            ++ewy;
        }
    }

    return Ax / (edgeWeightSumX * depthGradSumX) + Ay / (edgeWeightSumY * depthGradSumY);
}

cv::Mat DepthEdgeAlignment::InterpolateLidarPoints(const cv::Mat &srcLidar,
                                                   std::vector<ushort> &depthFromLidar,
                                                   std::vector<int>    &lidarPixelIndex,
                                                   cv::Mat             &reliableRegion,
                                                   int reliableRadius)
{
    cv::Rect imgBoundary(0, 0, srcLidar.cols, srcLidar.rows);
    
    cv::Subdiv2D subdiv(imgBoundary);

    for(int i = 0; i < srcLidar.rows; i++)
    {
        const double* p = srcLidar.ptr<double>(i);
        int index = i*srcLidar.cols;
        
        for(int j = 0; j < srcLidar.cols; j++)
        {
            if(*p > 0)
            {
                subdiv.insert(Point2f(j, i));
                
                depthFromLidar.emplace_back(*p);
                lidarPixelIndex.emplace_back(index);
                
                cv::circle(reliableRegion, Point(j, i), reliableRadius, cv::Scalar(255), -1);
            }

            ++p;
            ++index;
        }
    }
   
    cv::erode(reliableRegion, reliableRegion, getStructuringElement(cv::MORPH_ELLIPSE, Size(reliableRadius*2, reliableRadius*2)));
    

    cv::Mat srcLidarInterpolated = cv::Mat::zeros(srcLidar.size(), srcLidar.type());
    
    for(int i = 0; i < srcLidar.rows; i++)
    {
        const double* src = srcLidar.ptr<double>(i);
        double*       dst = srcLidarInterpolated.ptr<double>(i);

        for(int j = 0; j < srcLidar.cols; j++)
        {
            if(*src > 0)
            {
                *dst = *src;
            }
            else
            {
                // data is missing in src. Interepolate it from the neighbors
                Point2f targetPoint(j, i);
                Point2f neighborPoint[3];
                double  neighborDepth[3];
                
                int edgeId, vertexId;
                int pointLoc = subdiv.locate(targetPoint, edgeId, vertexId);

                for(int k = 0; k < 3; k++)
                {
                    neighborPoint[k] = subdiv.getVertex(subdiv.edgeOrg(edgeId));
                    
                    edgeId = subdiv.getEdge(edgeId, Subdiv2D::NEXT_AROUND_LEFT);
                }
                
                vector<bool> isPointValid(3, false);
                
                for(int k = 0; k < 3; ++k)
                {
                    if(imgBoundary.contains(neighborPoint[k]))
                    {
                        neighborDepth[k] = srcLidar.at<double>(neighborPoint[k].y, neighborPoint[k].x);
                        isPointValid[k]  = true;
                    }
                }
                
                bool isAllPointsValid = (isPointValid[0] && isPointValid[1] && isPointValid[2]);

                if(isAllPointsValid)
                {
                    float u, v, w;
                    Barycentric(targetPoint, neighborPoint[0], neighborPoint[1], neighborPoint[2], u, v, w);
                    
                    *dst = neighborDepth[0]*u + neighborDepth[1]*v + neighborDepth[2]*w;
                }
            }
            
            ++src;
            ++dst;
        }
    }
    
    return srcLidarInterpolated;
}

static cv::Mat GetLidarToCamera(cv::Vec6f &calibrationParams)
{
    cv::Mat lidarToCamera = cv::Mat::zeros(4, 4, CV_32F);
    
    cv::Vec6f &cp = calibrationParams;

    double length = sqrt(cp[0]*cp[0] + cp[1]*cp[1]);
    if(length >= 1.0)
    {
        cp[0] /= length;
        cp[1] /= length;
    }
    
    cv::Mat rotationAxis = (Mat_<float>(3, 1) << cp[0], cp[1], sqrt(1.0 - cp[0]*cp[0] - cp[1]*cp[1]));
    
    Rodrigues(rotationAxis * cp[2], lidarToCamera(cv::Rect(0, 0, 3, 3)));
    
    lidarToCamera.at<float>(0, 3) = cp[3];
    lidarToCamera.at<float>(1, 3) = cp[4];
    lidarToCamera.at<float>(2, 3) = cp[5];
    lidarToCamera.at<float>(3, 3) = 1.0f;

    return lidarToCamera;
}


void DepthEdgeAlignment::GetSmoothingWeights(double tau, double gamma)
{
    edgeWeightX = Mat::zeros(srcCamera.size(), CV_64F);
    edgeWeightY = Mat::zeros(srcCamera.size(), CV_64F);
    
    for(int y = 0; y < srcCamera.rows; y++)
    {
        const uchar*  p = srcCamera.ptr<uchar>(y);
        double*      wx = edgeWeightX.ptr<double>(y);
        
        for(int x = 0; x < srcCamera.cols - 1; x++)
        {
            *wx++ = *(p+1) - *p;
            ++p;
        }
    }
    
    for(int y = 0; y < srcCamera.rows - 1; y++)
    {
        const uchar* p1 = srcCamera.ptr<uchar>(y);
        const uchar* p2 = srcCamera.ptr<uchar>(y+1);
        double*      wy = edgeWeightY.ptr<double>(y);
        
        for(int x = 0; x < srcCamera.cols; x++)
        {
            *wy++ = *p2 - *p1;
            ++p1;
            ++p2;
        }
    }
    
    smoothingWeight = Mat::zeros(srcCamera.size(), CV_64F);
    
    CV_Assert(smoothingWeight.size() == edgeWeightX.size());
    CV_Assert(smoothingWeight.size() == edgeWeightY.size());
    
    for(int y = 0; y < smoothingWeight.rows; y++)
    {
        double* wx = edgeWeightX.ptr<double>(y);
        double* wy = edgeWeightY.ptr<double>(y);
        double*  w = smoothingWeight.ptr<double>(y);
        
        for(int x = 0; x < smoothingWeight.cols; x++)
        {
            *w++ = exp(-tau * sqrt((*wx) * (*wx) + (*wy) * (*wy)));
            
            *wx = exp(-gamma * fabs(*wx));
            *wy = exp(-gamma * fabs(*wy));
            
            ++wx;
            ++wy;
        }
    }
    
}


cv::Mat DepthEdgeAlignment::FindCalibrationParameters(const cv::Vec6f &calibrationParams,
                                                      const cv::Vec6f &neighborDelta)
{
    ofstream fout("iteration.csv");
    
    RNG rng;
    
    double minFidelity = std::numeric_limits<double>::max();
    cv::Vec6f bestCalibParams;
    
    cv::Mat depthMap;
    
    // run simulated-annealing
    const double minTemperature = 0.0007;
    const double alpha = 0.9;
    double temperature = 0.43;
    

    cv::Vec6f prevCalibrationParams = calibrationParams;
    double prevCalibrationFidelity  = DepthFusion(GetLidarToCamera(prevCalibrationParams),
                                                  false, // this disables edge-aware optimization
                                                  depthMap,
                                                  0);
    
    bestCalibParams = prevCalibrationParams;
    minFidelity     = prevCalibrationFidelity;
    
    int nIter = 1;
    
    while(temperature > minTemperature)
    {
        for(int k = 0; k < 10; ++k)
        {
            // update one of six calibration parameters per iteration
            int whichParam = rng.uniform(0, 6);
            
            cv::Vec6f newCalibrationParams = prevCalibrationParams;
            newCalibrationParams[whichParam] += neighborDelta[whichParam] * rng.uniform(-1.0f, 1.0f);
            
            double newCalibrationFidelity = DepthFusion(GetLidarToCamera(newCalibrationParams),
                                                        false, // this disables edge-aware optimization
                                                        depthMap,
                                                        nIter);
            
            if(newCalibrationFidelity < minFidelity)
            {
                minFidelity     = newCalibrationFidelity;
                bestCalibParams = newCalibrationParams;
            }
            
            cerr << "iter " << nIter << " cost " << newCalibrationFidelity << " " << newCalibrationParams[0] << " " << newCalibrationParams[1] << " " << newCalibrationParams[2]/M_PI*180.0
            << "\n" << GetLidarToCamera(newCalibrationParams) << endl;
            
            double costDelta = newCalibrationFidelity - prevCalibrationFidelity;
            
            double probability = exp(-costDelta*1000000 / temperature);
            double   randomVal = rng.uniform(0.0, 1.0);
 
#ifdef DEBUG
            cerr << costDelta << " " << probability << " " << randomVal << endl;
#endif
            
            if(costDelta < 0.0 || probability > randomVal)
            {
                cerr << "updated\n";
                prevCalibrationParams   = newCalibrationParams;
                prevCalibrationFidelity = newCalibrationFidelity;
            }
            
            fout << nIter << "," << prevCalibrationFidelity << ",";
            for(int i = 0; i < 6; i++)
            {
                fout << prevCalibrationParams[i] << ",";
            }
            fout << endl;
            
            nIter++;
        }
        
        temperature *= alpha;
    }
        
    cerr << "best calib params " << bestCalibParams << " min cost " << minFidelity << endl;
    
    cv::Mat bestLidarToCamera = GetLidarToCamera(bestCalibParams);
    
    cerr << "best LidarToCamera:\n";
    cerr << bestLidarToCamera << endl;
    
    fout.close();
    
    return bestLidarToCamera;
}

double DepthEdgeAlignment::DepthFusion(const cv::Mat &lidarToCamera,
                                       bool isAlignEdge,
                                       cv::Mat &optimizedDepth, // this is output
                                       int nIter)
{
    cv::Mat projectedLidarPoints = cv::Mat::zeros(srcCamera.size(), CV_64F);
    
#pragma omp parallel for
    for(size_t i = 0; i < lidarPoints.size(); ++i)
    {
        cv::Mat lidarCoord = (Mat_<float>(4, 1) << lidarPoints[i].x, lidarPoints[i].y, lidarPoints[i].z, 1);

        cv::Mat cameraCoord = lidarToCamera * lidarCoord;
        
        float depth = cameraCoord.at<float>(2);
        
        cv::Mat imageCoord = cameraToImage * cameraCoord;
        float invNormalize = 1.0 / imageCoord.at<float>(2);
        
        cv::Point normalizedImgCoord(imageCoord.at<float>(0) * invNormalize, imageCoord.at<float>(1) * invNormalize);
        
        if(cv::Rect(0, 0, srcCamera.cols, srcCamera.rows).contains(normalizedImgCoord) && depth > 0.0f)
        {
            projectedLidarPoints.at<double>(normalizedImgCoord.y, normalizedImgCoord.x) = depth;
        }
    }
    
#ifdef DEBUG
    imwrite("projectedLidarPoints_" + to_string(nIter) + ".png", ConvertToHeatmap(projectedLidarPoints, 80.0));
    imwrite("srcCamera.png", srcCamera);
#endif
    
    std::vector<ushort> depthFromLidar;
    
    // store the index of pixel (in 1D) from which the corresponding lidar data comes
    std::vector<int> lidarPixelIndex;
    
    // this is the area where we measure the error
    cv::Mat reliableRegion = cv::Mat::zeros(srcCamera.size(), CV_8U);
    
    cv::Mat srcLidarInterpolated = InterpolateLidarPoints(projectedLidarPoints,
                                                          depthFromLidar,
                                                          lidarPixelIndex,
                                                          reliableRegion,
                                                          5);

#ifdef DEBUG
    imwrite("srcLidarInterpolated_" + to_string(nIter) + ".png", ConvertToHeatmap(srcLidarInterpolated, 80.0));
    imwrite("reliableRegion_" + to_string(nIter) + ".png", reliableRegion);
#endif
    
    optimizedDepth = PrimalDualHybridGradient(srcLidarInterpolated, 2, depthFromLidar, lidarPixelIndex,
                                              isAlignEdge ? smoothingWeight : cv::Mat(), // disable smoothing weight during calibration
                                              isAlignEdge ? 300 : 100);

#ifdef DEBUG
    imwrite("optimized_" + to_string(nIter) + ".png", ConvertToHeatmap(optimizedDepth, 80));
#endif
    
    return EstimateCalibrationFidelity(optimizedDepth, reliableRegion);
}

int main(int argc, char ** argv)
{
    DepthEdgeAlignment edgeAlign;

    cv::Mat lidarToCamera; // This is the "true" extrinsic camera matrix
    
    edgeAlign.ReadData("image_2/",
                       "velodyne/",
                       "calib/",
                       "001665",//"002893"; // 003969 // 001665
                       lidarToCamera);

    
    // Set "reCalibrate" to true if you need to re-calculate calibration data.
    // The extrinsic camera matrix (=lidarToCamera) is updated using simulated-annealing
    bool reCalibrate = false;

    if(reCalibrate)
    {
        cv::Mat trueRotationVector;
        Rodrigues(lidarToCamera(cv::Rect(0, 0, 3, 3)), trueRotationVector);
        
        float trueAngle = sqrt(trueRotationVector.dot(trueRotationVector));
        trueRotationVector *= (1.0 / trueAngle);
        
        cerr << "true angle " << trueAngle/M_PI*180.0 << endl;
        cerr << "true rotation vector\n" << trueRotationVector << endl;
        cerr << "true translation\n" << lidarToCamera.at<float>(0, 3) << " " << lidarToCamera.at<float>(1, 3) << " " << lidarToCamera.at<float>(2, 3)  << endl;

        cv::Vec6f neighborDelta = { 0.001, 0.001, toRadian(3.0), 0.01, 0.01, 0.001 };
        
        // rot_axis_X, rot_axis_Y, angle, transX, transY, transZ
        cv::Vec6f initCalibrationParams(trueRotationVector.at<float>(0),
                                        trueRotationVector.at<float>(1),
                                        trueAngle,
                                        lidarToCamera.at<float>(0, 3),
                                        lidarToCamera.at<float>(1, 3),
                                        lidarToCamera.at<float>(2, 3));
        
        
        cv::Mat depthMap;
        cerr << "true fidelity " << edgeAlign.DepthFusion(GetLidarToCamera(initCalibrationParams),
                                                          false, // this disables edge-aware optimization
                                                          depthMap,
                                                          -1) << endl;
        
        // perturb calibration data for test
        cv::RNG rng;
        cv::Vec6f &cp = initCalibrationParams;
        
        for(int i = 0; i < 6; ++i)
        {
            cp[i] += neighborDelta[i] * rng.uniform(-1.0f, 1.0f) * 3.0;
        }
        
        lidarToCamera = edgeAlign.FindCalibrationParameters(cp, neighborDelta);
    }

    // edge-aware depth map optimization using "true/updated" calibration data
    cv::Mat edgeAlignedDepth;
    edgeAlign.DepthFusion(lidarToCamera, true, edgeAlignedDepth, 0);
    
    imwrite("edgeAlignedDepth.png", ConvertToHeatmap(edgeAlignedDepth, 80.0));
    
    return 0;
}
