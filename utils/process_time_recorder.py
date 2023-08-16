from math import floor
import time
import math



time_recorder=[]

def record_process(  process_name:str, process_timing:int,workpiece_id=None,view_id=None):
    '''
    workpiece_id:  
    view_id:    
    process_name:  crop or predict etc
    process_timing:    0:begin  10:end
    '''
    if workpiece_id is None:
        workpiece_id=""
    if view_id is None:
        view_id=""
    time_recorder.append({"workpiece_id":workpiece_id, 
                          "view_id":view_id,
                          "process_name":process_name,
                          "process_timing":process_timing,
                          "time_stamp":str(  int(time.time()*1000))
                          })
    

def reset_id(workpiece_id:str="",view_id:str=""):
    if len(time_recorder)>0:
        for i in range(len(time_recorder)):
            if time_recorder[i]["workpiece_id"]=="":
                time_recorder[i]["workpiece_id"]=workpiece_id
                time_recorder[i]["view_id"]=view_id
                

def pop_time_record():
    temp=time_recorder.copy()
    time_recorder.clear()
    return temp