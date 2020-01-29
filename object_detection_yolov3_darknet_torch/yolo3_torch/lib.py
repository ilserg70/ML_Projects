
from __future__ import unicode_literals, division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import os 
import os.path as osp
import pandas as pd
import random 
import pickle as pkl
import itertools
import re
import shutil
from functools import reduce
import math
from collections import Counter

# scikit-image==0.16.2
from skimage.measure import compare_ssim

# scipy==1.1.0
from scipy.optimize import linear_sum_assignment

# https://github.com/ytdl-org/youtube-dl/blob/master/README.md#output-template
# sudo -H pip3 install --upgrade youtube-dl
import youtube_dl

import urllib.request as url_req
import requests

from yolo3_torch.darknet import Darknet
from yolo3_torch.bbox import bbox_iou

def setObjId(val):
    global globalObjId
    globalObjId = 0.

def get_next_obj_id():
    global globalObjId
    globalObjId += 1.
    return globalObjId

def get_video_width_height(fileVideoInp):
    videoInp = cv2.VideoCapture(fileVideoInp)
    fw = int(videoInp.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(videoInp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoInp.release()
    return (fw, fh)

def get_params(frames_per_sec, frame_width, frame_height):
    return {
        'frame_width': float(frame_width),
        'frame_height': float(frame_height),
        'frame_diag': math.sqrt(frame_width**2 + frame_height**2),
        'frame_area': float(frame_width*frame_height),
        'frames_per_second': frames_per_sec,
        'time_step': 1./float(frames_per_sec),
        'similar_classes': {
            2: [7], # car => truck
            5: [7], # bus => truck
            7: [2,5]# truck => car, bus
        }
    }

def load_classes(namesFile):
    with open(namesFile, "r") as fp:
        return fp.read().split("\n")[:-1]

def upload_model(cfgFile, weightsFile, inpDim, CUDA):
    model = Darknet(cfgFile)
    model.load_weights(weightsFile)
    model.net_info["height"] = inpDim
    inpDim = int(model.net_info["height"])
    assert inpDim % 32 == 0 
    assert inpDim > 32
    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()
    model.eval() # set the model in evaluation mode
    return model

def get_sorted_files(dirPath):
    files = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
    files.sort(key = lambda f: int(re.search('[^0-9]*(\d+)\..*', f).group(1)))
    return [os.path.join(dirPath, f) for f in files]

def clean_and_mk_dir(dirPath):
    if dirPath:
        if os.path.exists(dirPath):
            shutil.rmtree(dirPath)
        os.makedirs(dirPath)

def find_files(dirPath, pattern):
    files = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
    ptrn = re.compile(pattern)
    files = list(filter(lambda f: ptrn.match(f), files))
    return [os.path.join(dirPath, f) for f in files]

def add_suffix(fileName, sfx):
    ff = os.path.splitext(fileName)
    return ff[0] + sfx + ff[1]

def get_file_name(fileName):
    return os.path.basename(fileName)

def get_name(fileName):
    return os.path.splitext(get_file_name(fileName))[0]

# # python-magic-bin==0.4.14
# import magic
# def get_file_type(file_path):
#     mime = magic.Magic(mime=True)
#     file_info = mime.from_file(file_path)
#     for t in ['video','image','application']:
#         if file_info.find(t) != -1:
#             return (file_info, t)
#     return (file_info, '')

def download_image(url):
    file_data = url_req.urlopen(url)
    data_to_write = file_data.read()
    image_buf = np.frombuffer(data_to_write, np.uint8)
    frame = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
    return frame

def download_from_youtube(url, destDir):
    id_ = url.split('/')[-1]
    ydl_opts = {
        'outtmpl': destDir + '/%(id)s.%(ext)s'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    files = find_files(destDir, '{}\..*'.format(id_))
    return files[0] if files else None

def download_videofile(url, file_name):
    url_req.urlretrieve(url, file_name)
    return file_name if os.path.exists(file_name) else None

def download_videofile2(url, file_name):
    r = requests.get(url, stream = True)
    with open(file_name, 'wb') as f: 
        for chunk in r.iter_content(chunk_size = 1024*1024): 
            if chunk: 
                f.write(chunk)
    return file_name if os.path.exists(file_name) else None

def letterbox_image(img, inpDim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inpDim
    k = min(w/img_w, h/img_h)
    new_w, new_h = int(img_w * k), int(img_h * k)
    resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
    canvas = np.full((h, w, 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    return canvas

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def get_results(prediction, confidence, numClasses, nms = True, nmsConf = 0.4):
    """
    Get the boxes with object confidence > threshold
    Convert the coordinates to absolute coordinates
    Perform NMS on these boxes, and save the results 
    I could have done NMS and saving seperately to have a better abstraction
    But both these operations require looping, hence 
    clubbing these ops in one loop instead of two. 
    loops are slower than vectorised operations.
    """
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return None

    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    
    batch_size = prediction.size(0)
    
    output = None

    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]

        # Get the class having maximum score, and the index of that class
        # Get rid of numClasses softmax scores 
        # Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ numClasses], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))

        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        
        # Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:,-1])
        except:
             continue
        
        # WE will do NMS classwise
        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

             # sort the detections such that the entry with the maximum objectness confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            # if nms has to be done
            if nms:
                # For each detection
                for i in range(idx):
                    # Get the IOUs of all boxes that come after the one we are looking at in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
                    except IndexError:
                        break
                    
                    # Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nmsConf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    # Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            # Concatenate the batch_id of the image to the detection
            # this helps us identify which image does the detection correspond to 
            # We use a linear straucture to hold ALL the detections from the batch
            # the batch_dim is flattened
            # batch is identified by extra batch column
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            out = torch.cat(seq,1)
            output = out if output is None else torch.cat((output, out))
    return output

def add_column_for_object_id(res):
    res2 = torch.FloatTensor(res.size(0), res.size(1)+1).fill_(0.)
    res2[:,:-2] = res[:,:-1]
    res2[:,-1] = res[:,-1]
    return res2

def image_to_model_input(origImg, newDim):
    '''Prepare image for inputting to the neural network. '''
    img = (letterbox_image(origImg, (newDim, newDim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def rescaling_to_original(detected, origDim, inpDim):
    im_dim_list = torch.FloatTensor(origDim).repeat(detected.shape[0],1)
    scaling_factor = torch.min(inpDim/im_dim_list,1)[0].view(-1,1)
    detected[:,[1,3]] -= (inpDim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    detected[:,[2,4]] -= (inpDim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    detected[:,1:5] /= scaling_factor
    for i in range(detected.shape[0]):
        detected[i, [1,3]] = torch.clamp(detected[i, [1,3]], 0.0, im_dim_list[i,0])
        detected[i, [2,4]] = torch.clamp(detected[i, [2,4]], 0.0, im_dim_list[i,1])
    return detected

def choose_color(colors, objId):
    return colors[objId%len(colors)] if objId!=0 else random.choice(colors)

def mark_one_object(frame, obj, colors, classes, name_len=None):
    objId = int(obj[-2])
    color = choose_color(colors, objId)
    p1 = tuple(obj[1:3].int()) # start point
    p2 = tuple(obj[3:5].int()) # end point
    name = classes[int(obj[-1])]
    if name_len is not None:
        name = name[:name_len]
    label = "{0}{1}".format(name, objId)
    cv2.rectangle(frame, p1, p2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c3 = p1[0] + t_size[0] + 3, p1[1] + t_size[1] + 4
    cv2.rectangle(frame, p1, c3, color, -1)
    cv2.putText(frame, label, (p1[0], c3[1]), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)

def visualize_objects(frameId, frm, results, outVideo, colors, classes, framesDir=None, outDir=None, name_len=None):
    if framesDir is not None and os.path.exists(framesDir):
        cv2.imwrite(os.path.join(framesDir, f"frame{int(frameId)}.jpg"), frm)
    res = results[(results[:,0]==frameId).nonzero().squeeze(1)]
    for obj in res:
        mark_one_object(frm, obj, colors, classes, name_len)
    outVideo.write(frm)
    if outDir is not None and os.path.exists(outDir):
        cv2.imwrite(os.path.join(outDir, f"frame{int(frameId)}.jpg"), frm)

def sub_img(frame, obj):
    x1, y1 = tuple(obj[1:3].int())
    x2, y2 = tuple(obj[3:5].int())
    return frame[y1:y2+1,x1:x2+1]
    
def center(o):
    return torch.tensor([o[1]+o[3], o[2]+o[4]])/2.
    
def width(o):
    return (o[3]-o[1]).abs()

def height(o):
    return (o[4]-o[2]).abs()

def area(o):
    return width(o) * height(o)

def size_rate(o):
    return width(o) / height(o)

def overlap_area(x0a, y0a, x1a, y1a, x0b, y0b, x1b, y1b):
    if x1a <= x0b or x1b <= x0a or y1a <= y0b or y1b <= y0a:
        return torch.tensor(0.)
    x1, x2 = max(x0a, x0b), min(x1a, x1b)
    y1, y2 = max(y0a, y0b), min(y1a, y1b)
    return (x2-x1)*(y2-y1)

def overlap(a, b):
    return 2.*overlap_area(a[1], a[2], a[3], a[4], b[1], b[2], b[3], b[4])/(area(a)+area(b))

def normalized_overlap(a, b, W, H, factor=0.33):
    ca, cb = center(a), center(b)
    s = torch.FloatTensor([W, H])*factor/2.
    na = torch.cat((a[:1], ca-s, ca+s),dim=0)
    nb = torch.cat((b[:1], cb-s, cb+s),dim=0)
    return overlap(na, nb)

def euclid_dist2(a, b):
    return ((center(b)-center(a))**2).sum()

def euclid_dist(a, b):
    return euclid_dist2(a, b).sqrt()

def time_btw_frames(a, b):
    return (a[0]-b[0]).abs()

def diff_rate(x,y):
    return 1.-2.*(x-y).abs()/(x+y)

def rvrs_exp(val):
    return 1./val.exp()

def ssim_score(imgA, imgB):
    if imgA.shape != imgB.shape:
        imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]), interpolation = cv2.INTER_CUBIC)
    imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    (ssim, diff) = compare_ssim(imgA, imgB, full=True)
    return ssim

def calc_match_metrics(a, b, params):
    """
    a = [frameId, x0, y0, x1, y1, conf., conf., objId, classId]
    """    
    distRate = euclid_dist(a, b)/params['frame_diag']
    distScore = 1.-distRate
    over = overlap(a, b)
    normOver = normalized_overlap(a, b, params['frame_width'], params['frame_height'], factor=0.33)
    #ssim = ssim_score(imgA, imgB)
    return [over, normOver, distScore]

def match_score(a, b, params, objMatchModel=None):
    if objMatchModel is None:
        metrics = calc_match_metrics(a, b, params)
        score = sum(metrics)/3.
        return score
    else:
        ff = torch.FloatTensor([metrics]).numpy()
        predict = objMatchModel.predict(ff)
        if predict==0. and score < 0.8:
            return torch.tensor(0.)
        probs = objMatchModel.predict_proba(ff)
        return probs[0][1]
        #return score if predict==1. else torch.tensor(0.)

def hungarian_algo(graph, isMax=True):
    '''return column indices'''
    row_ind, col_ind = linear_sum_assignment(-graph) if isMax else linear_sum_assignment(graph)
    #optim_val = graph[row_ind, col_ind].sum()
    return col_ind

def self_match_frame(res, params, objMatchModel=None):
    size = res.shape[0]
    for i in range(size-1):
        classA = int(res[i,-1])
        sc = params['similar_classes'].get(classA,{})
        for j in range(i+1,size):
            if res[j,-2] != 0.:
                continue
            classB = int(res[j,-1])
            if classB in sc:
                prob = match_score(res[i], res[j], params, objMatchModel)
                if prob != 0. and res[i,-2] != res[j,-2]:
                    if res[i,-2]==0. and res[j,-2]==0.:
                        res[i,-2] = res[j,-2] = get_next_obj_id()
                    elif res[i,-2]==0.:
                        res[i,-2] = res[j,-2]
                    elif res[j,-2]==0.:
                        res[j,-2] = res[i,-2]
                    elif res[i,-2]<res[j,-2]:
                        res[j,-2] = res[i,-2]
                    else:
                        res[i,-2] = res[j,-2]
    for obj in res:
        if obj[-2]==0.:
            obj[-2] = get_next_obj_id()

def match_two_frames(resA, resB, params, objMatchModel=None):
    lenA, lenB = resA.shape[0], resB.shape[0]
    size = max(lenA, lenB)
    graph = torch.zeros([size, size], dtype=torch.float32)
    for i, objA in enumerate(resA):
        classA = int(objA[-1])
        sc = params['similar_classes'].get(classA,{})
        for j, objB in enumerate(resB):
            classB, idB = int(objB[-1]), objB[-2]
            if idB==0. and (classA==classB or classB in sc):
                graph[i,j] = match_score(objA, objB, params, objMatchModel)
            
    b_ind = hungarian_algo(graph, isMax=True)
    for i in range(lenA):
        j = b_ind[i]
        if j < lenB and graph[i][j] > 0.:
            if resB[j,-2]==0. and (resA[i,-2] not in resB[:,-2]):
                resB[j,-2] = resA[i,-2]
    return (graph, b_ind)

def find_matches(cachFrm, params, results, objMatchModel=None):
    if not cachFrm:
        return    
    idB, _, resB = cachFrm[-1] # last frame
    for i in list(range(len(cachFrm)-1))[::-1]:
        if (resB[:,-2]==0.).sum(dim=0)!=0:
            idA, _, resA = cachFrm[i]
            match_two_frames(resA, resB, params, objMatchModel)
    self_match_frame(resB, params, objMatchModel)
    results[(results[:,0]==idB).nonzero().squeeze(1),-2] = resB[:,-2]
    
def restore_miss_object(objId, frameId, leftFrameId, rightFrameId, results):
    leftObj  = results[(results[:,0]==leftFrameId)  * (results[:,-2]==objId)]
    rightObj = results[(results[:,0]==rightFrameId) * (results[:,-2]==objId)]
    # [frameId, x0, y0, x1, y1, conf., conf., objId, classId]
    newObj = (leftObj*(rightFrameId-frameId) + rightObj*(frameId-leftFrameId))/(rightFrameId-leftFrameId)
    newObj[0,0] = frameId
    newObj[0,7:9] = leftObj[0,7:9] # objId, classId
    return newObj

def restore_miss_detection(cachFrm, results):
    if not cachFrm or len(cachFrm)<=1:
        return results
    restored = None
    frameId1, frameId2 = cachFrm[0][0], cachFrm[-1][0]
    #print(f"frameId1={frameId1}, frameId2={frameId2}")
    frameFilter = (results[:,0]>=frameId1) * (results[:,0]<=frameId2)
    #print(f"res = {results[frameFilter].int()}")
    objIds = np.unique(results[frameFilter,-2].numpy())
    #print(f"objIds={objIds}")
    #print(f"len(cachFrm)={len(cachFrm)}")

    for objId in objIds:
        frameIds = np.unique(results[(results[:,-2]==objId) * frameFilter,0].sort()[0].numpy())
        #print(f"objId={objId}")
        #print(f"\tframeIds={frameIds}")
        minId, maxId, size = frameIds[0], frameIds[-1], frameIds.shape[0]
        #print(f"\tminId={minId}, maxId={maxId}, size={size}")
        # If missed detection on some frames
        if int(maxId-minId+1) != size:
            #print(f"\tmaxId-minId+1={int(maxId-minId+1)} != size={size}")
            frameId = minId
            for k in range(size):
                while frameId != frameIds[k]:
                    #print(f"\tRestore object {int(objId)} on frame {int(frameId)} by frames ({int(frameIds[k-1])}, {int(frameIds[k])})")
                    newObj = restore_miss_object(objId, frameId, frameIds[k-1], frameIds[k], results)
                    restored = newObj if restored is None else torch.cat((restored, newObj))
                    frameId += 1
                frameId += 1
    return results if restored is None else torch.cat((results, restored))

def get_detected_classes(res, classes):
    return [classes[int(x[-1])] for x in res]

def logging_classes(frameId, res, classes):
    objs = get_detected_classes(res, classes)
    return "Frame: {0:6d} Detected: {1:s}".format(int(frameId), ", ".join(objs))


