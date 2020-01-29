import torch

from axon.ai.schemas import Message

class Transform(Message):
    vehicle = {'car','bus','truck','train','boat','motorbike','aeroplane'}
    monitor = {'tvmonitor','laptop','cell phone'}
    other = {'person','traffic light','stop sign','knife'}
    objects = vehicle | monitor | other
    idx = {k: i for i, k in enumerate(['frameNum','xmin','ymin','xmax','ymax','conf1','conf2','objId','objIdx'])}

    def get_inference_request_info(self, request) -> dict:
        return {'asset_id': request.get('assetId',''), 'request_id': request.get('requestId','')}

    def convert_to_detected_objects(self, results: torch.FloatTensor, classes: list) -> list:
        """
        :param results: torch.FloatTensor [frameId, xmin, ymin, xmax, ymax, conf., conf., objId, objClassId]
        :return: list of: {'xmin': int, 'ymin': int, 'xmax': int, 'ymax': int, 'class': str, 'frameId': int, 'objId': int, 'conf1': float, 'conf2': float}
        """
        return [{
            'xmin': int(r[self.idx['xmin']]),
            'ymin': int(r[self.idx['ymin']]),
            'xmax': int(r[self.idx['xmax']]),
            'ymax': int(r[self.idx['ymax']]),
            'class': classes[int(r[self.idx['objIdx']])]
        } for r in results] if results is not None else None

    def annotation_records_to_inference_response(self, annotation_records: list, inference_request: dict):
        """
        Transform InferenceRequest to InferenceResponse

        :param annotation_records:  - list of records of types: ['AVRAnnotationData','ALPRAnnotationData', ...]
        :param inference_request:   - record of type 'InferenceRequest'
        :return:         - list of records of type 'InferenceResponse'
        """

        tasks = {task['taskType']: task for task in inference_request.get('tasks',[])}
        annotations = {
            r['name']: self.get_schema(list(filter(lambda x: isinstance(x,dict), r['type']))[0].get('items',''))
            for r in self.get_schema('AnnotationResults').get('fields',[]) if r['name'] in tasks
         } # { 'AVR': schema('AVRAnnotationData'), 'ALPR': schema('ALPRAnnotationData), ... }

        responses = []
        schema_response = self.get_schema('InferenceResponse')
        for label, schema in annotations.items():
            features = [r for r in annotation_records if self.validate_precisely(r, schema)]
            resp = {k: inference_request.get(k,'') for k in ['requestId','assetId','dataType']}
            resp['task'] = tasks[label]
            resp['features'] = {label:  None}
            if len(features) > 0:
                resp['features'][label] = features            
            if self.validate_precisely(resp, schema_response):
                responses.append(resp)
        return responses

    def detected_objects_to_inference_response(self, taskType: str, detected_objects: list, inference_request: dict):
        """
        Transform detected objects to InferenceResponse

        :param taskType:          - one of: "ALPR", "AVR"
        :param detected_objects:  - list of objects, example: {'class': 'car', 'xmin': 20.4, 'ymin': 12.0, 'width': 114.1, 'height': 76.0}
        :param inference_request: - record of type 'InferenceRequest'
        :return:                  - record of type 'InferenceResponse'
        """

        def to_standard_box(box: dict):
            """
            :param box: - object of type dict
            :return: box with keys xmin, ymin, xmax, ymax, ...
            """
            if 'xmin' not in box:
                box['xmin'] = box.get('Xmin', box.get('x0', box.get('X0', None)))
            if 'ymin' not in box:
                box['ymin'] = box.get('Ymin', box.get('y0', box.get('Y0', None)))
            if box['xmin'] is None or box['ymin'] is None:
                return None
            if 'xmax' not in box:
                box['xmax'] = box['xmin'] + box.get('width', box.get('Width', box.get('w', box.get('W', None))))
            if 'ymax' not in box:
                box['ymax'] = box['ymin'] + box.get('height', box.get('Height', box.get('h', box.get('H', None))))
            if box['xmax'] is None or box['ymax'] is None:
                return None
            return box

        annotation_records = []
        if taskType == 'AVR':
            for box in detected_objects:
                box = to_standard_box(box)
                if box is not None:
                    annotation_records.append({
                        'vehicleType': box['class'],
                        'boxCoords': {
                            'topLeft':     {'x': box['xmin'], 'y': box['ymin']},
                            'bottomRight': {'x': box['xmax'], 'y': box['ymax']}
                        }
                    })
        elif taskType == 'ALPR':
            for box in detected_objects:
                box = to_standard_box(box)
                if box is not None:
                    annotation_records.append({
                        'licensePlateLocation': [
                            {'x': box['xmin'], 'y': box['ymin']},
                            {'x': box['xmax'], 'y': box['ymin']},
                            {'x': box['xmax'], 'y': box['ymax']},
                            {'x': box['xmin'], 'y': box['ymax']}
                        ]
                    })

        responses = self.annotation_records_to_inference_response(annotation_records, inference_request)
        return responses[0] if len(responses) > 0 else {}

    def records_to_ObjectTracks(self, results: torch.FloatTensor, classes: list) -> list:
        """
        Transform results to list of ObjectTrack

        :param results: ['frameNum','xmin','ymin','xmax','ymax','conf1','conf2','objId','objIdx']

        :return: ObjectTrack = {
            'trackId': 'int',
            'trackName': 'string',
            'objectType': 'string',
            'segments': {'_ARRAY': {
                'frameNum': 'int',
                'maskType': 'string',
                'maskSource': 'string',
                'boundingBox': {
                    'topLeft':     {'x': 'double', 'y': 'double'},
                    'bottomRight': {'x': 'double', 'y': 'double'}}}},
            'alprMeta': {
                'licensePlateState': 'string', 
                'licensePlateText': 'string'},
            'avrMeta': {
                'vehicleType': 'string',
                'vehicleSubType': 'string',
                'vehicleMake': 'string',
                'vehicleModel': 'string',
                'vehicleColor': 'string'},
            'faceMeta': {
                'pose': 'string',
                'occlusion': 'string',
                'gender': 'string',
                'ethnicity': 'string'},
            'mdtMeta': {
                'typeClass': 'string', 
                'isOn': 'boolean'},
            'customMetadata': {'_MAP': 'string'}}
        """

        minObjId = int(results[:,self.idx['objId']].min())
        maxObjId = int(results[:,self.idx['objId']].max())

        tracks = []
        for objId_ in range(minObjId, maxObjId+1):
            rr = results[(results[:,self.idx['objId']]==objId_).nonzero().squeeze(1)]
            if rr.shape[0] > 0:
                objClass = classes[int(rr[0,self.idx['objIdx']])]
                if objClass not in self.objects:
                    continue
                objectTrack = {
                    'trackId': objId_,
                    'trackName': objClass,
                    'objectType': objClass,
                    'segments': [{
                        'frameNum': int(r[self.idx['frameNum']]),
                        'maskType': 'BoundingBox',
                        'maskSource': 'preprocessing',
                        'boundingBox': {
                            'topLeft': {'x': int(r[self.idx['xmin']]), 'y': int(r[self.idx['ymin']])},
                            'bottomRight': {'x': int(r[self.idx['xmax']]), 'y': int(r[self.idx['ymax']])}
                        }
                    } for r in rr],
                    'alprMeta': None,
                    'avrMeta': None,
                    'faceMeta': None,
                    'mdtMeta': None,
                    'customMetadata': None
                }
                if objClass in self.vehicle:
                    objectTrack['avrMeta'] = {
                        'vehicleType': objClass,
                        'vehicleSubType': None,
                        'vehicleMake': None,
                        'vehicleModel': None,
                        'vehicleColor': None
                    }
                elif objClass in self.monitor:
                    objectTrack['mdtMeta'] = {
                        'typeClass': objClass, 
                        'isOn': False
                    }
                elif objClass in self.other:
                    objectTrack['customMetadata'] = {
                        'objectType': objClass
                    }
                tracks.append(objectTrack)

        return tracks