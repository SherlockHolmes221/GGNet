import json
import numpy as np

class hico():
    def __init__(self, annotation_file,out_name):
        self.annotations = json.load(open(annotation_file, 'r'))
        self.out_name = out_name
        self.train_annotations = json.load(open(annotation_file.replace('test_hico.json', 'trainval_hico.json'),'r'))
        self.overlap_iou = 0.5
        self.verb_name_dict =[[1, 5, 5], [1, 5, 18], [1, 5, 26], [1, 5, 31], [1, 5, 42], [1, 5, 53], [1, 5, 77], [1, 5, 88], [1, 5, 112], [1, 5, 58], [1, 2, 9], [1, 2, 37], [1, 2, 42], [1, 2, 44], [1, 2, 38], [1, 2, 63], [1, 2, 72], [1, 2, 76], [1, 2, 77], [1, 2, 88], [1, 2, 99], [1, 2, 111], [1, 2, 112], [1, 2, 58], [1, 16, 11], [1, 16, 27], [1, 16, 37], [1, 16, 66], [1, 16, 75], [1, 16, 113], [1, 16, 58], [1, 9, 5], [1, 9, 22], [1, 9, 26], [1, 9, 42], [1, 9, 44], [1, 9, 48], [1, 9, 76], [1, 9, 77], [1, 9, 78], [1, 9, 80], [1, 9, 88], [1, 9, 94], [1, 9, 106], [1, 9, 112], [1, 9, 58], [1, 44, 9], [1, 44, 21], [1, 44, 37], [1, 44, 42], [1, 44, 49], [1, 44, 59], [1, 44, 70], [1, 44, 58], [1, 6, 5], [1, 6, 18], [1, 6, 22], [1, 6, 26], [1, 6, 42], [1, 6, 53], [1, 6, 77], [1, 6, 88], [1, 6, 112], [1, 6, 114], [1, 6, 58], [1, 3, 5], [1, 3, 18], [1, 3, 22], [1, 3, 39], [1, 3, 42], [1, 3, 44], [1, 3, 53], [1, 3, 63], [1, 3, 77], [1, 3, 112], [1, 3, 58], [1, 17, 23], [1, 17, 27], [1, 17, 37], [1, 17, 40], [1, 17, 46], [1, 17, 66], [1, 17, 81], [1, 17, 112], [1, 17, 11], [1, 17, 58], [1, 62, 9], [1, 62, 37], [1, 62, 50], [1, 62, 88], [1, 62, 94], [1, 62, 58], [1, 63, 9], [1, 63, 50], [1, 63, 88], [1, 63, 58], [1, 21, 27], [1, 21, 35], [1, 21, 37], [1, 21, 40], [1, 21, 46], [1, 21, 47], [1, 21, 56], [1, 21, 66], [1, 21, 77], [1, 21, 111], [1, 21, 58], [1, 67, 13], [1, 67, 25], [1, 67, 87], [1, 67, 58], [1, 18, 9], [1, 18, 23], [1, 18, 27], [1, 18, 34], [1, 18, 37], [1, 18, 39], [1, 18, 40], [1, 18, 42], [1, 18, 46], [1, 18, 66], [1, 18, 79], [1, 18, 81], [1, 18, 99], [1, 18, 108], [1, 18, 111], [1, 18, 112], [1, 18, 11], [1, 18, 58], [1, 19, 27], [1, 19, 34], [1, 19, 37], [1, 19, 40], [1, 19, 44], [1, 19, 46], [1, 19, 53], [1, 19, 38], [1, 19, 66], [1, 19, 73], [1, 19, 77], [1, 19, 79], [1, 19, 99], [1, 19, 108], [1, 19, 111], [1, 19, 112], [1, 19, 58], [1, 4, 37], [1, 4, 42], [1, 4, 44], [1, 4, 38], [1, 4, 63], [1, 4, 72], [1, 4, 73], [1, 4, 77], [1, 4, 88], [1, 4, 99], [1, 4, 109], [1, 4, 111], [1, 4, 112], [1, 4, 58], [1, 1, 9], [1, 1, 32], [1, 1, 37], [1, 1, 40], [1, 1, 46], [1, 1, 93], [1, 1, 101], [1, 1, 103], [1, 1, 49], [1, 1, 58], [1, 64, 9], [1, 64, 37], [1, 64, 39], [1, 64, 58], [1, 20, 9], [1, 20, 27], [1, 20, 35], [1, 20, 37], [1, 20, 40], [1, 20, 46], [1, 20, 66], [1, 20, 77], [1, 20, 84], [1, 20, 111], [1, 20, 112], [1, 20, 58], [1, 7, 5], [1, 7, 22], [1, 7, 26], [1, 7, 53], [1, 7, 77], [1, 7, 88], [1, 7, 112], [1, 7, 58], [1, 72, 14], [1, 72, 76], [1, 72, 113], [1, 72, 58], [1, 53, 8], [1, 53, 16], [1, 53, 24], [1, 53, 37], [1, 53, 42], [1, 53, 65], [1, 53, 67], [1, 53, 90], [1, 53, 112], [1, 53, 58], [1, 27, 9], [1, 27, 37], [1, 27, 42], [1, 27, 59], [1, 27, 115], [1, 27, 58], [1, 52, 8], [1, 52, 9], [1, 52, 16], [1, 52, 24], [1, 52, 37], [1, 52, 42], [1, 52, 65], [1, 52, 67], [1, 52, 90], [1, 52, 58], [1, 39, 6], [1, 39, 9], [1, 39, 37], [1, 39, 85], [1, 39, 100], [1, 39, 105], [1, 39, 116], [1, 39, 58], [1, 40, 37], [1, 40, 115], [1, 40, 58], [1, 23, 27], [1, 23, 41], [1, 23, 113], [1, 23, 58], [1, 65, 13], [1, 65, 50], [1, 65, 88], [1, 65, 58], [1, 15, 42], [1, 15, 50], [1, 15, 88], [1, 15, 58], [1, 84, 9], [1, 84, 37], [1, 84, 59], [1, 84, 74], [1, 84, 58], [1, 51, 37], [1, 51, 97], [1, 51, 112], [1, 51, 49], [1, 51, 58], [1, 56, 16], [1, 56, 24], [1, 56, 37], [1, 56, 90], [1, 56, 97], [1, 56, 112], [1, 56, 58], [1, 61, 4], [1, 61, 9], [1, 61, 16], [1, 61, 24], [1, 61, 37], [1, 61, 52], [1, 61, 55], [1, 61, 68], [1, 61, 58], [1, 57, 9], [1, 57, 15], [1, 57, 16], [1, 57, 24], [1, 57, 37], [1, 57, 65], [1, 57, 90], [1, 57, 97], [1, 57, 112], [1, 57, 58], [1, 77, 9], [1, 77, 37], [1, 77, 74], [1, 77, 76], [1, 77, 102], [1, 77, 104], [1, 77, 58], [1, 85, 12], [1, 85, 37], [1, 85, 76], [1, 85, 83], [1, 85, 58], [1, 47, 9], [1, 47, 21], [1, 47, 37], [1, 47, 42], [1, 47, 70], [1, 47, 86], [1, 47, 90], [1, 47, 28], [1, 47, 112], [1, 47, 58], [1, 60, 8], [1, 60, 9], [1, 60, 24], [1, 60, 37], [1, 60, 55], [1, 60, 68], [1, 60, 90], [1, 60, 58], [1, 22, 27], [1, 22, 37], [1, 22, 39], [1, 22, 40], [1, 22, 46], [1, 22, 38], [1, 22, 66], [1, 22, 77], [1, 22, 111], [1, 22, 112], [1, 22, 113], [1, 22, 58], [1, 11, 40], [1, 11, 42], [1, 11, 59], [1, 11, 62], [1, 11, 58], [1, 48, 37], [1, 48, 51], [1, 48, 96], [1, 48, 49], [1, 48, 112], [1, 48, 58], [1, 34, 3], [1, 34, 10], [1, 34, 37], [1, 34, 91], [1, 34, 105], [1, 34, 58], [1, 25, 27], [1, 25, 46], [1, 25, 66], [1, 25, 77], [1, 25, 113], [1, 25, 58], [1, 89, 37], [1, 89, 60], [1, 89, 76], [1, 89, 58], [1, 31, 9], [1, 31, 37], [1, 31, 42], [1, 31, 58], [1, 58, 9], [1, 58, 15], [1, 58, 16], [1, 58, 24], [1, 58, 37], [1, 58, 55], [1, 58, 58], [1, 76, 9], [1, 76, 13], [1, 76, 37], [1, 76, 110], [1, 76, 58], [1, 38, 2], [1, 38, 9], [1, 38, 31], [1, 38, 37], [1, 38, 42], [1, 38, 48], [1, 38, 71], [1, 38, 58], [1, 49, 17], [1, 49, 37], [1, 49, 96], [1, 49, 112], [1, 49, 116], [1, 49, 49], [1, 49, 58], [1, 73, 37], [1, 73, 59], [1, 73, 74], [1, 73, 76], [1, 73, 110], [1, 73, 58], [1, 78, 13], [1, 78, 59], [1, 78, 60], [1, 78, 58], [1, 74, 14], [1, 74, 37], [1, 74, 76], [1, 74, 58], [1, 55, 8], [1, 55, 16], [1, 55, 24], [1, 55, 37], [1, 55, 42], [1, 55, 65], [1, 55, 67], [1, 55, 92], [1, 55, 112], [1, 55, 58], [1, 79, 13], [1, 79, 37], [1, 79, 42], [1, 79, 59], [1, 79, 76], [1, 79, 60], [1, 79, 58], [1, 14, 12], [1, 14, 64], [1, 14, 76], [1, 14, 58], [1, 59, 8], [1, 59, 9], [1, 59, 15], [1, 59, 16], [1, 59, 24], [1, 59, 37], [1, 59, 55], [1, 59, 68], [1, 59, 89], [1, 59, 90], [1, 59, 58], [1, 82, 13], [1, 82, 37], [1, 82, 57], [1, 82, 59], [1, 82, 58], [1, 75, 37], [1, 75, 69], [1, 75, 100], [1, 75, 58], [1, 54, 9], [1, 54, 15], [1, 54, 16], [1, 54, 24], [1, 54, 37], [1, 54, 55], [1, 54, 58], [1, 87, 17], [1, 87, 37], [1, 87, 59], [1, 87, 58], [1, 81, 13], [1, 81, 76], [1, 81, 112], [1, 81, 58], [1, 41, 9], [1, 41, 29], [1, 41, 33], [1, 41, 37], [1, 41, 44], [1, 41, 68], [1, 41, 77], [1, 41, 88], [1, 41, 94], [1, 41, 58], [1, 35, 1], [1, 35, 9], [1, 35, 37], [1, 35, 42], [1, 35, 44], [1, 35, 68], [1, 35, 76], [1, 35, 77], [1, 35, 94], [1, 35, 115], [1, 35, 58], [1, 36, 1], [1, 36, 9], [1, 36, 33], [1, 36, 37], [1, 36, 44], [1, 36, 77], [1, 36, 94], [1, 36, 115], [1, 36, 58], [1, 50, 37], [1, 50, 49], [1, 50, 112], [1, 50, 86], [1, 50, 58], [1, 37, 3], [1, 37, 9], [1, 37, 10], [1, 37, 20], [1, 37, 36], [1, 37, 37], [1, 37, 42], [1, 37, 45], [1, 37, 68], [1, 37, 82], [1, 37, 85], [1, 37, 91], [1, 37, 105], [1, 37, 58], [1, 13, 37], [1, 13, 95], [1, 13, 98], [1, 13, 58], [1, 33, 9], [1, 33, 19], [1, 33, 37], [1, 33, 40], [1, 33, 53], [1, 33, 59], [1, 33, 61], [1, 33, 68], [1, 33, 117], [1, 33, 58], [1, 42, 9], [1, 42, 19], [1, 42, 37], [1, 42, 42], [1, 42, 44], [1, 42, 50], [1, 42, 53], [1, 42, 77], [1, 42, 94], [1, 42, 88], [1, 42, 112], [1, 42, 58], [1, 88, 9], [1, 88, 37], [1, 88, 40], [1, 88, 46], [1, 88, 58], [1, 43, 9], [1, 43, 37], [1, 43, 42], [1, 43, 100], [1, 43, 58], [1, 32, 1], [1, 32, 16], [1, 32, 37], [1, 32, 42], [1, 32, 71], [1, 32, 106], [1, 32, 115], [1, 32, 58], [1, 80, 37], [1, 80, 60], [1, 80, 76], [1, 80, 58], [1, 70, 13], [1, 70, 30], [1, 70, 59], [1, 70, 76], [1, 70, 88], [1, 70, 94], [1, 70, 112], [1, 70, 58], [1, 90, 7], [1, 90, 37], [1, 90, 112], [1, 90, 58], [1, 10, 43], [1, 10, 76], [1, 10, 95], [1, 10, 98], [1, 10, 58], [1, 8, 18], [1, 8, 22], [1, 8, 42], [1, 8, 53], [1, 8, 76], [1, 8, 77], [1, 8, 88], [1, 8, 112], [1, 8, 58], [1, 28, 9], [1, 28, 37], [1, 28, 54], [1, 28, 59], [1, 28, 76], [1, 28, 83], [1, 28, 95], [1, 28, 58], [1, 86, 37], [1, 86, 55], [1, 86, 62], [1, 86, 58], [1, 46, 28], [1, 46, 37], [1, 46, 86], [1, 46, 107], [1, 46, 49], [1, 46, 112], [1, 46, 58], [1, 24, 27], [1, 24, 37], [1, 24, 66], [1, 24, 113], [1, 24, 58]]

        self.fp = {}
        self.tp = {}
        self.score = {}
        self.sum_gt = {}
        self.file_name = []
        self.train_sum = {}
        for gt_i in self.annotations:
            self.file_name.append(gt_i['file_name'])
            gt_hoi = gt_i['hoi_annotation']
            gt_bbox = gt_i['annotations']
            for gt_hoi_i in gt_hoi:
                if isinstance(gt_hoi_i['category_id'], str):
                    gt_hoi_i['category_id'] = int(gt_hoi_i['category_id'].replace('\n', ''))
                triplet = [gt_bbox[gt_hoi_i['subject_id']]['category_id'],gt_bbox[gt_hoi_i['object_id']]['category_id'],gt_hoi_i['category_id']]
                if triplet not in self.verb_name_dict:
                    assert False
                if self.verb_name_dict.index(triplet) not in self.sum_gt.keys():
                    self.sum_gt[self.verb_name_dict.index(triplet)] =0
                self.sum_gt[self.verb_name_dict.index(triplet)] += 1
        for train_i in self.train_annotations:
            train_hoi = train_i['hoi_annotation']
            train_bbox = train_i['annotations']
            for train_hoi_i in train_hoi:
                if isinstance(train_hoi_i['category_id'], str):
                    train_hoi_i['category_id'] = int(train_hoi_i['category_id'].replace('\n', ''))
                triplet = [train_bbox[train_hoi_i['subject_id']]['category_id'],train_bbox[train_hoi_i['object_id']]['category_id'],train_hoi_i['category_id']]
                if triplet not in self.verb_name_dict:
                    continue
                if self.verb_name_dict.index(triplet) not in self.train_sum.keys():
                    self.train_sum[self.verb_name_dict.index(triplet)] =0
                self.train_sum[self.verb_name_dict.index(triplet)] += 1
        for i in range(len(self.verb_name_dict)):
            self.fp[i] = []
            self.tp[i] = []
            self.score[i] = []
        self.r_inds = []
        self.c_inds = []
        for id in self.train_sum.keys():
            if self.train_sum[id] < 10:
                self.r_inds.append(id)
            else:
                self.c_inds.append(id)

        self.num_class = len(self.verb_name_dict)

    def evalution(self, predict_annot):
        for pred_i in predict_annot:
            if pred_i['file_name'] not in self.file_name:
                continue
            gt_i = self.annotations[self.file_name.index(pred_i['file_name'])]
            gt_bbox = gt_i['annotations']
            if len(gt_bbox)!=0:
                pred_bbox = self.add_One(pred_i['predictions']) #convert zero-based to one-based indices
                if len(pred_bbox) == 0:
                    print(pred_i['file_name'])
                    continue
                bbox_pairs, bbox_ov = self.compute_iou_mat(gt_bbox, pred_bbox)
                pred_hoi = pred_i['hoi_prediction']
                gt_hoi = gt_i['hoi_annotation']
                self.compute_fptp(pred_hoi, gt_hoi, bbox_pairs, pred_bbox,bbox_ov)
            else:
                pred_bbox = self.add_One(pred_i['predictions']) #convert zero-based to one-based indices
                for i, pred_hoi_i in enumerate(pred_i['hoi_prediction']):
                    triplet = [pred_bbox[pred_hoi_i['subject_id']]['category_id'],
                               pred_bbox[pred_hoi_i['object_id']]['category_id'], pred_hoi_i['category_id']]
                    verb_id = self.verb_name_dict.index(triplet)
                    self.tp[verb_id].append(0)
                    self.fp[verb_id].append(1)
                    self.score[verb_id].append(pred_hoi_i['score'])
        map = self.compute_map()
        return map

    def compute_map(self):
        output_txt = self.out_name + '.txt'
        f = open(output_txt, 'a')

        ap = np.zeros(self.num_class)
        max_recall = np.zeros(self.num_class)
        for i in range(len(self.verb_name_dict)):
            sum_gt = self.sum_gt[i]

            if sum_gt == 0:
                continue
            tp = np.asarray((self.tp[i]).copy())
            fp = np.asarray((self.fp[i]).copy())
            res_num = len(tp)
            if res_num == 0:
                continue
            score = np.asarray(self.score[i].copy())
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gt
            prec = tp / (fp + tp)
            ap[i] = self.voc_ap(rec,prec)
            max_recall[i] = np.max(rec)
            info = 'hoi_id: {}, ap: {}   max recall: {}'.format(i+1, self.verb_name_dict[i], ap[i], max_recall[i])
            # print(info)
            f.write(info)
            f.write('\n')

        mAP = np.mean(ap[:])
        mAP_rare = np.mean(ap[self.r_inds])
        mAP_nonrare = np.mean(ap[self.c_inds])
        m_rec = np.mean(max_recall[:])
        print('--------------------')
        print('mAP: {} mAP rare: {}  mAP nonrare: {}  max recall: {}'.format(mAP, mAP_rare, mAP_nonrare, m_rec))
        print('--------------------')
        f.write('mAP: {} mAP rare: {}  mAP nonrare: {}  max recall: {}'.format(mAP, mAP_rare, mAP_nonrare, m_rec))
        f.write('mAP: {}   max recall: {}'.format(mAP, m_rec))
        f.write('---------------------------------------------\n')
        return mAP

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_hoi, gt_hoi, match_pairs, pred_bbox,bbox_ov):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hoi))
        pred_hoi.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hoi) != 0:
            for i, pred_hoi_i in enumerate(pred_hoi):
                is_match = 0
                if isinstance(pred_hoi_i['category_id'], str):
                    pred_hoi_i['category_id'] = int(pred_hoi_i['category_id'].replace('\n', ''))
                if len(match_pairs) != 0 and pred_hoi_i['subject_id'] in pos_pred_ids and pred_hoi_i['object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi_i['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi_i['object_id']]
                    pred_obj_ov=bbox_ov[pred_hoi_i['object_id']]
                    pred_sub_ov=bbox_ov[pred_hoi_i['subject_id']]
                    pred_category_id = pred_hoi_i['category_id']
                    max_ov=0
                    max_gt_id=0
                    for gt_id in range(len(gt_hoi)):
                        gt_hoi_i = gt_hoi[gt_id]
                        if (gt_hoi_i['subject_id'] in pred_sub_ids) and (gt_hoi_i['object_id'] in pred_obj_ids) and (pred_category_id == gt_hoi_i['category_id']):
                            is_match = 1
                            min_ov_gt=min(pred_sub_ov[pred_sub_ids.index(gt_hoi_i['subject_id'])], pred_obj_ov[pred_obj_ids.index(gt_hoi_i['object_id'])])
                            if min_ov_gt>max_ov:
                                max_ov=min_ov_gt
                                max_gt_id=gt_id
                if pred_hoi_i['category_id'] not in list(self.fp.keys()):
                    continue
                triplet = [pred_bbox[pred_hoi_i['subject_id']]['category_id'], pred_bbox[pred_hoi_i['object_id']]['category_id'], pred_hoi_i['category_id']]
                if triplet not in self.verb_name_dict:
                    continue
                verb_id = self.verb_name_dict.index(triplet)
                if is_match == 1 and vis_tag[max_gt_id] == 0:
                    self.fp[verb_id].append(0)
                    self.tp[verb_id].append(1)
                    vis_tag[max_gt_id] =1
                else:
                    self.fp[verb_id].append(1)
                    self.tp[verb_id].append(0)
                self.score[verb_id].append(pred_hoi_i['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0:
            assert False
        if len(bbox_list2) == 0:
            assert False
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i

        iou_mat_ov=iou_mat.copy()
        iou_mat[iou_mat>= 0.5] = 1
        iou_mat[iou_mat< 0.5] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pairs_ov={}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pairs_ov[pred_id]=[]
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pairs_ov[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id])
        return match_pairs_dict,match_pairs_ov

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
            S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
                return intersect / (sum_area - intersect)
        else:
            return 0

    def add_One(self,prediction):  #Add 1 to all coordinates
        for i, pred_bbox in enumerate(prediction):
            rec = pred_bbox['bbox']
            rec[0]+=1
            rec[1]+=1
            rec[2]+=1
            rec[3]+=1
        return prediction

