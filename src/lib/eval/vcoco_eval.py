import json
import numpy as np
import os
import sys, argparse
sys.path.insert(0, "/home/xian/Documents/code/GGNet/src/lib")

from eval.vsrl_eval import VCOCOeval


class vcoco():
    def __init__(self, annotation_file, out_name):
        self.annotations = json.load(open(annotation_file, 'r'))
        self.out_name = out_name
        self.overlap_iou = 0.5
        self.verb_name_list = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26,
                               28]
        self.fp = {}
        self.tp = {}
        self.score = {}
        self.sum_gt = {}
        self.verb_name_dict = {2: 'cut_instr', 21: 'snowboard_instr', 4: 'cut_obj', 0: 'surf_instr',
                               26: 'skateboard_instr', 7: 'kick_obj', 9: 'eat_obj', 14: 'carry_obj', 15: 'throw_obj',
                               16: 'eat_instr', 17: 'smile', 18: 'look_obj', 19: 'hit_instr', 20: 'hit_obj',
                               1: 'ski_instr', 22: 'run', 10: 'sit_instr', 24: 'read_obj', 5: 'ride_instr', 3: 'walk',
                               23: 'point_instr', 11: 'jump_instr', 8: 'work_on_computer_instr', 25: 'hold_obj',
                               13: 'drink_instr', 12: 'lay_instr', 6: 'talk_on_phone_instr', 27: 'stand',
                               28: 'catch_obj'}
        for i in self.verb_name_list:
            self.fp[i] = []
            self.tp[i] = []
            self.score[i] = []
            self.sum_gt[i] = 0
        self.file_name = []
        for gt_i in self.annotations:
            self.file_name.append(gt_i['file_name'])
            gt_hoi = gt_i['hoi_annotation']
            for gt_hoi_i in gt_hoi:
                if isinstance(gt_hoi_i['category_id'], str):
                    gt_hoi_i['category_id'] = int(gt_hoi_i['category_id'].replace('\n', ''))
                if gt_hoi_i['category_id'] in self.verb_name_list:
                    self.sum_gt[gt_hoi_i['category_id']] += 1
        self.num_class = len(self.verb_name_list)

    def evalution(self, predict_annot):
        for pred_i in predict_annot:
            if pred_i['file_name'] not in self.file_name:
                continue
            gt_i = self.annotations[self.file_name.index(pred_i['file_name'])]
            gt_bbox = gt_i['annotations']
            if 'predictions' in pred_i.keys():
                pred_bbox = pred_i['predictions']
            elif 'predcition' in pred_i.keys():
                pred_bbox = pred_i['prediction']
            elif 'annotations' in pred_i.keys():
                pred_bbox = pred_i['annotations']
            elif 'annotation' in pred_i.keys():
                pred_bbox = pred_i['annotation']
            else:
                print('prediction file keys error')
            if 'hoi_prediction' in pred_i.keys():
                pred_hoi = pred_i['hoi_prediction']
            elif 'hoi_predictions' in pred_i.keys():
                pred_hoi = pred_i['hoi_predictions']
            elif 'hoi_annotation' in pred_i.keys():
                pred_hoi = pred_i['hoi_annotation']
            else:
                print('prediction file keys error')
            gt_hoi = gt_i['hoi_annotation']
            bbox_pairs = self.compute_iou_mat(gt_bbox, pred_bbox)
            self.compute_fptp(pred_hoi, gt_hoi, bbox_pairs)
        map = self.compute_map()
        return map

    def compute_map(self):
        ap = np.zeros(self.num_class)
        max_recall = np.zeros(self.num_class)
        output_txt = self.out_name + '.txt'
        f = open(output_txt, 'a')
        for i in range(len(self.verb_name_list)):
            sum_gt = self.sum_gt[self.verb_name_list[i]]

            if sum_gt == 0:
                continue
            tp = np.asarray((self.tp[self.verb_name_list[i]]).copy())
            fp = np.asarray((self.fp[self.verb_name_list[i]]).copy())
            res_num = len(tp)
            if res_num == 0:
                continue
            score = np.asarray(self.score[self.verb_name_list[i]].copy())
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gt
            prec = tp / (fp + tp)
            ap[i] = self.voc_ap(rec, prec)
            max_recall[i] = np.max(rec)

            info = 'class {} --- ap: {}   max recall: {}  pos: {}'.format(self.verb_name_dict[self.verb_name_list[i]],
                                                                          ap[i], max_recall[i], sum_gt)
            f.write(info)
            f.write('\n')

        mAP = np.mean(ap[:])
        mAP = (mAP - ap[20]) * 25 / 24
        m_rec = np.mean(max_recall[:])
        print('--------------------')
        print('mAP: {}   max recall: {}'.format(mAP, m_rec))
        print('--------------------')
        f.write('mAP: {}   max recall: {}'.format(mAP, m_rec))
        f.write('---------------------------------------------\n')
        return mAP

    def voc_ap(self, rec, prec):
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def compute_fptp(self, pred_hoi, gt_hoi, match_pairs):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hoi))
        pred_hoi.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hoi) != 0:
            for i, pred_hoi_i in enumerate(pred_hoi):
                is_match = 0
                if isinstance(pred_hoi_i['category_id'], str):
                    pred_hoi_i['category_id'] = int(pred_hoi_i['category_id'].replace('\n', ''))
                if len(match_pairs) != 0 and pred_hoi_i['subject_id'] in pos_pred_ids and pred_hoi_i[
                    'object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi_i['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi_i['object_id']]
                    pred_category_id = pred_hoi_i['category_id']
                    for gt_id in np.nonzero(1 - vis_tag)[0]:
                        gt_hoi_i = gt_hoi[gt_id]
                        if (gt_hoi_i['subject_id'] in pred_sub_ids) and (gt_hoi_i['object_id'] in pred_obj_ids) and (
                                pred_category_id == gt_hoi_i['category_id']):
                            is_match = 1
                            vis_tag[gt_id] = 1
                            continue
                if pred_hoi_i['category_id'] not in list(self.fp.keys()):
                    continue
                if is_match == 1:
                    self.fp[pred_hoi_i['category_id']].append(0)
                    self.tp[pred_hoi_i['category_id']].append(1)

                else:
                    self.fp[pred_hoi_i['category_id']].append(1)
                    self.tp[pred_hoi_i['category_id']].append(0)
                self.score[pred_hoi_i['category_id']].append(pred_hoi_i['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i
        iou_mat[iou_mat >= self.overlap_iou] = 1
        iou_mat[iou_mat < self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                match_pairs_dict[pred_id].append(match_pairs[0][i])
        return match_pairs_dict

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))

        rec1 = bbox1['bbox']
        rec2 = bbox2['bbox']
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

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
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return intersect / (sum_area - intersect)

    def evalution_offical(self, predict_annot):
        # [{'image_id':  # the coco image id,
        #   'person_box':  # [x1, y1, x2, y2] the box prediction for the person,
        # '[action]_agent':  # the score for action corresponding to the person prediction,
        # '[action]_[role]':  # [x1, y1, x2, y2, s], the predicted box for role and
        # # associated score for the action-role pair.
        # }]
        print(len(predict_annot))
        pred_det = []
        for pred in predict_annot:
            file_name = pred['file_name']
            image_id = int(file_name[:-4].split("_")[-1])
            predictions = pred['predictions']
            hoi_prediction = pred['hoi_prediction']
            pred_this_img = []
            for hoi in hoi_prediction:
                subject_id = hoi['subject_id']
                object_id = hoi['object_id']
                category_id = hoi['category_id']
                score = hoi['score']
                verb_role_name = self.verb_name_dict[category_id]
                role = verb_role_name.split("_")[-1]
                verb = verb_role_name[:-(len(role) + 1)]
                person_box = predictions[subject_id]['bbox']
                object_box = predictions[object_id]['bbox']
                find = False
                for p in pred_this_img:
                    if p['person_box'] == person_box:
                        if f"{verb}_{role}" in p:
                            if p[f"{verb}_{role}"][0:4] == object_box:
                                score_ = max(score, p[f"{verb}_{role}"][-1])
                                p[f"{verb}_{role}"] = [object_box[0], object_box[1],
                                                       object_box[2], object_box[3], score_]
                                find = True
                                break

                        else:
                            p[f"{verb}_{role}"] = [object_box[0], object_box[1],
                                                   object_box[2], object_box[3], score]
                            find = True
                            break
                if not find:
                    pred_this_img.append({
                        "image_id": image_id,
                        "person_box": person_box,
                        f"{verb}_{role}": [object_box[0], object_box[1],
                                           object_box[2], object_box[3], score]
                    })
            pred_det.extend(pred_this_img)

        vcocoeval = VCOCOeval("/home/xian/Documents/code/GGNet/Dataset/verbcoco/annotations/vcoco_test.json",
                              "/home/xian/Documents/code/GGNet/Dataset/verbcoco/annotations/instances_vcoco_all_2014.json",
                              "/home/xian/Documents/code/GGNet/Dataset/verbcoco/annotations/vcoco_test.ids")
        vcocoeval._do_eval(pred_det, ovr_thresh=0.5, out_name=self.out_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp',
        default="hoidet_vcoco_ggnet",
        type=str)
    parser.add_argument(
        '--start_epoch',
        default=100,
        type=int)
    parser.add_argument(
        '--end_epoch',
        default=120,
        type=int)
    args = parser.parse_args()
    dir = args.exp
    for num in range(args.start_epoch, args.end_epoch+1):
        vcoco_off = vcoco("/home/xian/Documents/code/GGNet/verbcoco/annotations/test_vcoco.json",
                          f"/home/xian/Documents/code/GGNet/exp/hoidet/{dir}/predictions_model_{num}")
        vcoco_off.evalution_offical(json.load(
            open(
                f"/home/xian/Documents/code/GGNet/exp/hoidet/{dir}/predictions_model_{num}.json",
                "r")))
