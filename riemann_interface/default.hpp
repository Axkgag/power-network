
class json;
class string;
namespace cv{
    class Mat;
}
/**
 * @brief """推理类基类
    需要实现3个接口：
    setupPredictor(self,config:Dict[str,Any])
    loadModel(self,model_dir_path:str)->bool
    predict(self,img:Any, info:Dict[str,V]=None)->Dict[str,Any]
 * 
 */
class AbstractPredictor{

    AbstractPredictor(json config);

    /**
     * @brief 加载参数，进行createmodel 等过程
     * 
     * @param config 
     */
    virtual void setupPredictor(json config)=0;

    /**
     * @brief 从单个文件夹加载predictor所有需要的文件，如果config 与当前模型有结构差异，需要重建model(可调用setupPredictor)
        
     * 
     * @param model_dir_path 文件夹路径，一般使用绝对路径
     * @return true             加载成功
     * @return false            加载失败
     */
    virtual bool loadModel(string model_dir_path)=0;

    /**
     * @brief ""实际预测，读入图片和信息（可选），返回结果 
     * 
     * @param img   图片路径，在函数内判断
     * @param info  信息，如点位信息等，以字典形式传入
     * @return json 返回预测结果，以字典形式传出
     */
    virtual json predict(string img, json info)=0;

    /**
     * @brief ""实际预测，读入图片和信息（可选），返回结果  重载
     * 
     * @param img   cv 图片 ，在函数内判断
     * @param info  信息，如点位信息等，以字典形式传入
     * @return json 返回预测结果，以字典形式传出
     */
    virtual json predict(cv::Mat img, json info)=0;

};

    
    