# AbstractPredictor & AbstractTrainer v1

## Basic 
```python
class AbstractPredictor():
    """推理类基类
    需要实现3个接口：
    setupPredictor(self,config:Dict[str,Any])
    loadModel(self,model_dir_path:str)->bool
    predict(self,img:Any, info:Dict[str,V]=None)->Dict[str,Any]
    可选： stackPredict(self,img_list:List[Any], info_list:List[Dict[str,Any]]=None)->List[Dict[str,Any]]
    """
    def __init__(self, config:Dict[str,Any]=None):
        pass
    
    @abstractmethod
    def setupPredictor(self,config:Dict[str,Any]):
        """加载参数，进行createmodel 等过程
        """
        raise NotImplementedError
    
    @abstractmethod
    def loadModel(self,model_dir_path:str)->bool:
        """从单个文件夹加载predictor所有需要的文件，如果config 与当前模型有结构差异，需要重建model(可调用setupPredictor)
        :model_dir_path:    文件夹路径，一般使用绝对路径
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self,img:Any, info:Dict[str,Any]=None)->Dict[str,Any]:
        """实际预测，读入图片和信息（可选），返回结果
        :img:       图片或图片路径，在函数内判断
        :info:      信息，如点位信息等，以字典形式传入
        :return:    返回预测结果，以字典形式传出
        """
        raise NotImplementedError

    
    def stackPredict(self,img_list:List[Any], info_list:List[Dict[str,Any]]=None)->List[Dict[str,Any]]:
        result=[]
        for i in range(len(img_list)):
            img=img_list[i]
            info=None 
            if i<len(info_list):
                info=info_list[i]
            result.append(self.predict(img=img,info=info))
        return result
```

```c++
#include <string.h>
#include<opencv2/opencv.hpp>

/**
 * @brief """推理类基类
    需要实现3个接口：
    setupPredictor(self,config:Dict[str,Any])
    loadModel(self,model_dir_path:str)->bool
    predict(self,img:Any, info:Dict[str,V]=None)->Dict[str,Any]
 * 
 */

class AbstractPredictor{

    public:
        AbstractPredictor(){};
        virtual ~AbstractPredictor(){};
        /**
        * @brief 加载参数，进行createmodel 等过程
        * 
        * @param config 
        */
        virtual void setupPredictor(std::string config)=0;

        /**
        * @brief 从单个文件夹加载predictor所有需要的文件，如果config 与当前模型有结构差异，需要重建model(可调用setupPredictor)
            
        * 
        * @param model_dir_path 文件夹路径，一般使用绝对路径
        * @return true             加载成功
        * @return false            加载失败
        */
        virtual bool loadModel(std::string model_dir_path)=0;

        /**
        * @brief ""实际预测，读入图片和信息（可选），返回结果 
        * 
        * @param img   cv 图片 ，在函数内判断
        * @param info  信息，如点位信息等，以字典形式传入
        * @return Json 返回预测结果，以字典形式传出(转成string传出)
        */
        virtual std::string predict(cv::Mat* img, std::string part_info="") = 0;

        /**
        * @brief ""实际预测，读入图片和信息（可选），返回结果 重载
        * 
        * @param img_path   图片路径，在函数内判断
        * @param info  信息，如点位信息等，以字典形式传入
        * @return Json 返回预测结果，以字典形式传出(转成string传出)
        */
        virtual std::string predict(std::string img_path, std::string part_info="") = 0;

};
```


```python
class AbstractTrainer:
    """训练类基类
    需要实现3个接口：
    setupTrain(self,train_dir:str,is_resume:bool)->Status
    trainEpoch(self)->TrainStatus
    exportModel(self, dirpath:str)
    """
    def __init__(self,sender:AbstractSender):
        """初始化，建立训练状态、训练线程等
        """
        pass
  
    @abstractmethod
    def setupTrain(self,train_dir:str,is_resume:bool)->Status:
        """从训练文件夹初始化所有训练参数，包括建立训练模型，在训练时设定是否从上次训练继续
        :train_dir: 训练文件夹
        :is_resume: 是否从上次训练继续,如果为True 
        """
        raise NotImplemented

    @abstractmethod
    def trainEpoch(self,epoch:int)->TrainStatus:
        """单个epoch训练, 所有参数需要通过self的属性传入
        """
        raise NotImplemented
    
    @abstractmethod
    def save_tmp_model(self):
        raise NotImplemented
    
    @abstractmethod
    def load_tmp_model(self):
        raise NotImplemented

    def actualTrain(self)->None:
        # print('in')
        """在实际训练线程中进行的训练, 所有参数需要通过self的属性传入，其中start_epoch 和 end_epoch 需要在setupTrain中 完成 
        """
        pass
            
    def beginTrain(self,train_dir:str,is_resume:bool=False)->bool:
        """外部调用的接口，开始训练，初始化训练文件夹和训练状态，启动训练线程
        :train_dir:     训练文件夹
        :is_resume:     是否从上次训练继续
        """
        return True

    def resumeTrain(self)->bool:
        """继续训练
        """
        if not self.train_dir:
            return False
        return self.beginTrain(self.train_dir,is_resume=True)
        
    def endTrain(self)->bool:
        """结束训练：如果有已保存权重，直接停止线程，否则通过stop signal 在下个epoch 停止
        """
        return True
    
    def pauseTrain(self)->bool:
        """结束训练：如果有已保存权重，直接停止线程，否则通过stop signal 在下个epoch 停止
        """
        return True
        
    @abstractmethod
    def exportModel(self, export_path:str)->bool:
        """导出所有需要的文件到 export_path,确保在 predictor load model 时可用
        """
        raise NotImplemented
```


## TaskBased
```python
class AbstractObjectDetectPredictor(AbstractPredictor):
    """目标检测推理基类
    """
    def setCropConfig(self,config:Dict)->bool:
        key_list=["crop_w","crop_h","stride_x","stride_y"]
        for key in key_list:
            if (not key in config)  or type(key)!=type(int):
                return False
        self.crop_w=config["crop_w"]
        self.crop_h=config["crop_h"]
        self.stride_x=config["stride_x"]
        self.stride_y=config["stride_y"]
        return True

    @abstractmethod
    def setupPredictor(self,config:Dict[str,Any]):
        """加载参数，进行createmodel 等过程
        """
        self.use_patch=self.setCropConfig(config)
        return 
        
    def patchPredict(self,img:Any,  info:Dict[str,Any]=None):
        """patchPredict 滑窗检测

        Args:
            img (Any): opencv image
            info (Dict[str,Any], optional): 信息. Defaults to None.

        Returns:
            result (Dict): 与正常predict一样形式的检测结果
        """
        result_list=[]
        for crop_img in crops:
            result_list.append(crop_start+self.predict(crop_img))
        return postprocess(result_list)
    
    def patchPredict(self,img:os.PathLike,  info:Dict[str,Any]=None):
         """patchPredict 滑窗检测

        Args:
            img (Any): 图片文件路径
            info (Dict[str,Any], optional): 信息. Defaults to None.

        Returns:
            result (Dict): 与正常predict一样形式的检测结果
        """
        pass
    def smartPredict(self,img:Any,  info:Dict[str,Any]=None):
        if self.use_patch:
            self.patchPredict()
        else :
            self.predict()
```

```c++
/**
 * @brief """目标检测推理类基类
    包含2个公共接口
    bool setCropConfig(std::string config); 设置滑窗检测参数
    std::string patchPredict(cv::Mat* img, std::string part_info=""); 滑窗检测
    注意保证在setupPredictor 接口中调用 setCropConfig， loadmodel过程中调用 setupPredictor
 * 
 */
class AbstractObjectDetectPredictor: public AbstractPredictor{

    public:
        AbstractObjectDetectPredictor(){};
        virtual ~AbstractObjectDetectPredictor(){};
        /**
        * @brief 设置滑窗检测参数
        * 
        * @param config 
        */
        bool setCropConfig(std::string config);

        virtual bool setupPredictor(std::string config){ return setCropConfig(config);}

        /**
        * @brief ""滑窗检测，读入图片和信息（可选），返回结果 
        * 
        * @param img   cv 图片 ，在函数内判断
        * @param info  信息，如点位信息等，以字典形式传入
        * @return Json 返回预测结果，以字典形式传出(转成string传出)
        */
        std::string patchPredict(cv::Mat* img, std::string part_info="");

        /**
        * @brief ""滑窗检测，读入图片和信息（可选），返回结果 
        * 
        * @param img_path   图片路径
        * @param info  信息，如点位信息等，以字典形式传入
        * @return Json 返回预测结果，以字典形式传出(转成string传出)
        */
        std::string patchPredict(std::string img_path, std::string part_info="") ;

};
```

```python
class AbstractObjectDetectTrainer:
    """目标检测训练类基类
    需要实现3个接口：
    setupTrain(self,train_dir:str,is_resume:bool)->Status
    trainEpoch(self)->TrainStatus
    exportModel(self, dirpath:str)
    """
    def __init__(self,sender:AbstractSender):
        """初始化，建立训练状态、训练线程等
        """
        pass
  
    @abstractmethod
    def setupTrain(self,train_dir:str,is_resume:bool)->Status:
        """从训练文件夹初始化所有训练参数，包括建立训练模型，在训练时设定是否从上次训练继续，保证从config中读取 滑窗训练相关参数
        :train_dir: 训练文件夹
        :is_resume: 是否从上次训练继续,如果为True 
        """
        raise NotImplemented

    @abstractmethod
    def trainEpoch(self,epoch:int)->TrainStatus:
        """单个epoch训练, 所有参数需要通过self的属性传入，若启用了滑窗检测训练模式，要做对应的修改
        """
        raise NotImplemented
    
```

