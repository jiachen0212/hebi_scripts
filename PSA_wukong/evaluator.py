import numbers
from abc import ABCMeta
from collections import OrderedDict
from typing import Dict, List

import numpy as np
from SMore_core.evaluation.evaluation_builder import EVALUATORS
from SMore_core.evaluation.evaluator_base import EvaluatorBase
from SMore_core.utils.common import (all_gather, distributed, is_main_process,
                                     synchronize)
from SMore_core.utils.config import merge_dict

from SMore_seg.common.constants import (SegmentationInputsConstants,
                                        SegmentationModelOutputConstants)
from SMore_seg.default_config.evaluation_defaults import EvaluatorDefaults
import ipdb


class SegmentationEvaluatorBase(EvaluatorBase, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = merge_dict(
            EvaluatorDefaults.SegmentationEvaluatorBase_cfg, self.kwargs)
        self.num_classes = self.kwargs.get('num_classes')
        self.label_map = self.kwargs.get('label_map')
        self.ignore_label = self.kwargs.get('ignore_label')


@EVALUATORS.register_module()
class PixelBasedEvaluator(SegmentationEvaluatorBase):
    """
    以每个像素为例，计算以像素为单位的recall/precision.
    """
    HIT = 'hit'
    MISS = 'miss'
    FA = 'fa'
    PREDICT = 'predict'
    TARGET = 'target'
    NUM_IMAGES = 'num_images'

    DEFAULT_CONFIG = EvaluatorDefaults.PixelBasedEvaluator_cfg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = merge_dict(EvaluatorDefaults.PixelBasedEvaluator_cfg,
                                 self.kwargs)
        self.epsilon = self.kwargs.get('epsilon')
        self.head_id = self.kwargs.get('head_id')
        self.predict = np.array([0] * self.num_classes)
        self.target = np.array([0] * self.num_classes)
        self.hit = np.array([0] * self.num_classes)
        self.miss = np.array([0] * self.num_classes)
        self.fa = np.array([0] * self.num_classes)

    def single_process(self, inputs: Dict, outputs: Dict, **kwargs):
        pass

    @staticmethod
    def get_info_by_head_id(data, head_id):
        if head_id is None:
            return data
        else:
            return data[head_id]

    def process(self, inputs_list, outputs_list, **kwargs):
        w_list = [
            self.get_info_by_head_id(outputs_list[idy][SegmentationModelOutputConstants.PREDICT_SCORES],
                                     self.head_id).shape[-1]
            for idy in range(len(outputs_list))
        ]
        h_list = [
            self.get_info_by_head_id(outputs_list[idy][SegmentationModelOutputConstants.PREDICT_SCORES],
                                     self.head_id).shape[-2]
            for idy in range(len(outputs_list))
        ]

        def accumulate(batch_predict, batch_target):
            batch_predict = batch_predict.argmax(axis=1).astype(np.int)
            batch_target = batch_target.astype(np.int)

            batch_predict[batch_target ==
                          self.ignore_label] = self.ignore_label
            k = (batch_target >= 0) & (batch_target < self.num_classes)

            hist_info = np.bincount(
                self.num_classes * batch_target[k] + batch_predict[k],
                minlength=self.num_classes ** 2).reshape(self.num_classes,
                                                         self.num_classes)

            predict_s = hist_info.sum(0)
            target_s = hist_info.sum(1)
            hit = np.diag(hist_info)
            miss = target_s - hit
            fa = predict_s - hit
            self.predict += predict_s
            self.target += target_s
            self.hit += hit
            self.miss += miss
            self.fa += fa

        if min(w_list) != max(w_list) or min(h_list) != max(h_list):
            for idy in range(len(inputs_list)):
                predict = self.get_info_by_head_id(outputs_list[idy][SegmentationModelOutputConstants.PREDICT_SCORES],
                                                   self.head_id)[None, :]
                target = self.get_info_by_head_id(inputs_list[idy][SegmentationInputsConstants.TARGET],
                                                  self.head_id)[None, :]
                if SegmentationInputsConstants.WEIGHTS in inputs_list[idy]:
                    if inputs_list[idy][SegmentationInputsConstants.WEIGHTS] == 0:
                        continue
                if SegmentationInputsConstants.HEAD_WEIGHTS in inputs_list[idy]:
                    if inputs_list[idy][SegmentationInputsConstants.HEAD_WEIGHTS][self.head_id] == 0:
                        continue
                accumulate(predict, target)
        else:
            predict = np.concatenate([
                self.get_info_by_head_id(outputs_list[idy][SegmentationModelOutputConstants.PREDICT_SCORES],
                                         self.head_id)[None, :]
                for idy in range(len(outputs_list))
            ])
            target = np.concatenate([
                self.get_info_by_head_id(inputs_list[idy][SegmentationInputsConstants.TARGET],
                                         self.head_id)[None, :]
                for idy in range(len(inputs_list))
            ])
            weights = []
            for idy in range(len(inputs_list)):
                if SegmentationInputsConstants.WEIGHTS in inputs_list[idy]:
                    if inputs_list[idy][SegmentationInputsConstants.WEIGHTS] == 0:
                        weights.append(0)
                        continue
                if SegmentationInputsConstants.HEAD_WEIGHTS in inputs_list[idy]:
                    if inputs_list[idy][SegmentationInputsConstants.HEAD_WEIGHTS][self.head_id] == 0:
                        weights.append(0)
                        continue
                weights.append(1)
            weights = np.array(weights)
            predict = predict[weights == 1, ...]
            target = target[weights == 1, ...]
            if target.shape[0] != 0 and predict.shape[0] != 0:
                accumulate(predict, target)

    def reset(self):
        self.predict = np.array([0] * self.num_classes)
        self.target = np.array([0] * self.num_classes)
        self.hit = np.array([0] * self.num_classes)
        self.miss = np.array([0] * self.num_classes)
        self.fa = np.array([0] * self.num_classes)

    def calculate_iou(self, inputs):
        results = {}
        iou = inputs[PixelBasedEvaluator.HIT] / (inputs[PixelBasedEvaluator.HIT] + inputs[PixelBasedEvaluator.MISS]
                                                 + inputs[PixelBasedEvaluator.FA]) * 100

        for i, class_name in zip(range(self.num_classes), self.label_map):
            results[class_name] = iou[i]

        results['mean'] = np.nanmean(iou)
        return results

    def calculate_precision(self, inputs):
        results = {}

        precision = inputs[PixelBasedEvaluator.HIT] / (inputs[PixelBasedEvaluator.PREDICT]) * 100

        precision[precision > 100] = 0.

        for i, class_name in zip(range(self.num_classes), self.label_map):
            results[class_name] = precision[i]

        results['mean'] = np.nanmean(precision)
        return results

    def calculate_recall(self, inputs):
        results = {}

        recall = inputs[PixelBasedEvaluator.HIT] / (inputs[PixelBasedEvaluator.TARGET]) * 100
        recall[recall > 100] = 0.

        for i, class_name in zip(range(self.num_classes), self.label_map):
            results[class_name] = recall[i]

        results['mean'] = np.nanmean(recall)
        return results

    def evaluate(self):
        if distributed():
            synchronize()

            predict_list = all_gather(self.predict)
            target_list = all_gather(self.target)
            hit_list = all_gather(self.hit)
            miss_list = all_gather(self.miss)
            fa_list = all_gather(self.fa)
            if not is_main_process():
                return

            self.predict = np.array([0] * self.num_classes)
            self.target = np.array([0] * self.num_classes)
            self.hit = np.array([0] * self.num_classes)
            self.miss = np.array([0] * self.num_classes)
            self.fa = np.array([0] * self.num_classes)
            for each in predict_list:
                self.predict += each
            for each in target_list:
                self.target += each
            for each in hit_list:
                self.hit += each
            for each in miss_list:
                self.miss += each
            for each in fa_list:
                self.fa += each

        inputs = {
            PixelBasedEvaluator.HIT: self.hit,
            PixelBasedEvaluator.MISS: self.miss,
            PixelBasedEvaluator.FA: self.fa,
            PixelBasedEvaluator.PREDICT: self.predict,
            PixelBasedEvaluator.TARGET: self.target
        }
        result = {
            'iou': self.calculate_iou(inputs),
            'recall': self.calculate_recall(inputs),
            'precision': self.calculate_precision(inputs),
        }
        if self.head_id is not None:
            results = {'head-{}'.format(self.head_id): result}
            results = OrderedDict(results)
        else:
            results = OrderedDict(result)
        return results


@EVALUATORS.register_module()
class ImageBasedEvaluator(SegmentationEvaluatorBase):
    """
    以图片为单位计算Recall/Precision。
    """
    HIT = 'hit'
    MISS = 'miss'
    FA = 'fa'
    PREDICT = 'predict'
    TARGET = 'target'
    NUM_IMAGES = 'num_images'
    DEFAULT_CONFIG = EvaluatorDefaults.ImageBasedEvaluator_cfg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = merge_dict(EvaluatorDefaults.ImageBasedEvaluator_cfg,
                                 self.kwargs)
        self.score_threshold = self.kwargs.get('score_threshold')
        if self.score_threshold is None:
            self.score_threshold = [0] * self.num_classes
        self.iou_threshold = self.kwargs.get('iou_threshold')
        if isinstance(self.iou_threshold, numbers.Number):
            self.iou_threshold = [self.iou_threshold] * self.num_classes
        self.epsilon = self.kwargs.get('epsilon')

        self.num_images = np.array([0])
        self.predict = np.array([0] * self.num_classes)
        self.target = np.array([0] * self.num_classes)
        self.hit = np.array([0] * self.num_classes)
        self.miss = np.array([0] * self.num_classes)
        self.fa = np.array([0] * self.num_classes)
        self.cls = np.array(range(self.num_classes)).reshape(
            1, self.num_classes, 1)

    def single_process(self, input: Dict, output: Dict, **kwargs):
        pass

    def process(self, inputs_list: List[Dict], outputs_list: List[Dict],
                **kwargs):
        w_list = [
            outputs_list[idy][
                SegmentationModelOutputConstants.PREDICT_SCORES].shape[-1]
            for idy in range(len(outputs_list))
        ]
        h_list = [
            outputs_list[idy][
                SegmentationModelOutputConstants.PREDICT_SCORES].shape[-2]
            for idy in range(len(outputs_list))
        ]

        def accumulate(batch_predict, batch_target):
            batch = batch_predict.shape[0]
            self.num_images[0] += batch

            predict_score = batch_predict.reshape(batch, self.num_classes,
                                                  -1)  # N x C x HW
            predict_cls = predict_score.argmax(axis=1).astype(np.int)  # N x HW
            batch_target = batch_target.astype(np.int).reshape(batch,
                                                               -1)  # N x HW
            predict_cls[batch_target == self.ignore_label] = self.ignore_label

            score_valid = (predict_score > np.asarray(
                self.score_threshold).reshape(1, self.num_classes, 1)
                           )  # N x C x HW
            predict_cls = np.repeat(predict_cls.reshape(batch, 1, -1),
                                    self.num_classes,
                                    axis=1) == self.cls
            predict_cls &= score_valid
            target = np.repeat(batch_target.reshape(batch, 1, -1),
                               self.num_classes,
                               axis=1) == self.cls  # N x C xHW

            self.target += (target.sum(2) > 0).sum(0)
            self.predict += (predict_cls.sum(2) > 0).sum(0)
            iou = (target & predict_cls).sum(2) / (
                    self.epsilon + (target | predict_cls).sum(2))  # N x C

            self.hit += (iou > self.iou_threshold).sum(0)
            self.miss += ((iou < self.iou_threshold) & target.sum(2) >
                          0).sum(0)
            self.fa += ((iou < self.iou_threshold) & predict_cls.sum(2) >
                        0).sum(0)

        if min(w_list) != max(w_list) or min(h_list) != max(h_list):
            for idy in range(len(inputs_list)):
                predict = outputs_list[idy][
                              SegmentationModelOutputConstants.PREDICT_SCORES][None, :]
                target = inputs_list[idy][SegmentationInputsConstants.TARGET][
                         None, :]
                accumulate(predict, target)
        else:
            # TODO(zxyan): evaluate may slow.
            predicts = np.concatenate([
                outputs_list[idy][
                    SegmentationModelOutputConstants.PREDICT_SCORES][None, :]
                for idy in range(len(outputs_list))
            ])
            targets = np.concatenate([
                inputs_list[idy][SegmentationInputsConstants.TARGET][None, :]
                for idy in range(len(inputs_list))
            ])
            weights = []
            for idy in range(len(inputs_list)):
                if SegmentationInputsConstants.WEIGHTS in inputs_list[idy]:
                    if inputs_list[idy][SegmentationInputsConstants.WEIGHTS] == 0:
                        weights.append(0)
                        continue
                if SegmentationInputsConstants.HEAD_WEIGHTS in inputs_list[idy]:
                    if inputs_list[idy][SegmentationInputsConstants.HEAD_WEIGHTS][self.head_id] == 0:
                        weights.append(0)
                        continue
                weights.append(1)
            weights = np.array(weights)
            predicts = predicts[weights == 1, ...]
            targets = targets[weights == 1, ...]
            if targets.shape[0] != 0 and predicts.shape[0] != 0:
                accumulate(predicts, targets)

    def reset(self):
        self.num_images = np.array([0])
        self.predict = np.array([0] * self.num_classes)
        self.target = np.array([0] * self.num_classes)
        self.hit = np.array([0] * self.num_classes)
        self.miss = np.array([0] * self.num_classes)
        self.fa = np.array([0] * self.num_classes)

    def calculate_false_alarm(self, inputs):
        results = {}
        for i, class_name in zip(range(self.num_classes), self.label_map):
            results[class_name] = inputs[ImageBasedEvaluator.FA][i] / inputs[
                ImageBasedEvaluator.NUM_IMAGES][0]
        return results

    def calculate_recall(self, inputs):
        results = {}

        recall = inputs[ImageBasedEvaluator.HIT] / (
                inputs[ImageBasedEvaluator.TARGET] + self.epsilon) * 100
        recall[recall > 100] = 0.

        for i, class_name in zip(range(self.num_classes), self.label_map):
            results[class_name] = recall[i]

        results['mean'] = np.nanmean(recall)
        return results

    def calculate_precision(self, inputs):
        results = {}

        precision = inputs[ImageBasedEvaluator.HIT] / (
                inputs[ImageBasedEvaluator.PREDICT] + self.epsilon) * 100
        precision[precision > 100] = 0.

        for i, class_name in zip(range(self.num_classes), self.label_map):
            results[class_name] = precision[i]

        results['mean'] = np.nanmean(precision)
        return results

    def evaluate(self):
        if distributed():
            synchronize()

            num_images_list = all_gather(self.num_images)
            predict_list = all_gather(self.predict)
            target_list = all_gather(self.target)
            hit_list = all_gather(self.hit)
            miss_list = all_gather(self.miss)
            fa_list = all_gather(self.fa)
            if not is_main_process():
                return

            self.num_images = np.array([0])
            self.predict = np.array([0] * self.num_classes)
            self.target = np.array([0] * self.num_classes)
            self.hit = np.array([0] * self.num_classes)
            self.miss = np.array([0] * self.num_classes)
            self.fa = np.array([0] * self.num_classes)
            for each in num_images_list:
                self.num_images += each
            for each in predict_list:
                self.predict += each
            for each in target_list:
                self.target += each
            for each in hit_list:
                self.hit += each
            for each in miss_list:
                self.miss += each
            for each in fa_list:
                self.fa += each

        result = {}
        inputs = {
            ImageBasedEvaluator.HIT: self.hit,
            ImageBasedEvaluator.MISS: self.miss,
            ImageBasedEvaluator.FA: self.fa,
            ImageBasedEvaluator.PREDICT: self.predict,
            ImageBasedEvaluator.TARGET: self.target,
            ImageBasedEvaluator.NUM_IMAGES: self.num_images
        }
        result['recall'] = self.calculate_recall(inputs)
        result['precision'] = self.calculate_precision(inputs)
        result['false_alarm'] = self.calculate_false_alarm(inputs)
        result['score_threshold'] = {}
        for i in range(self.num_classes):
            result['score_threshold'][
                self.label_map[i]] = self.score_threshold[i]
        results = OrderedDict(result)
        return results


@EVALUATORS.register_module()
class KillRateRelatedEvaluator(SegmentationEvaluatorBase):
    """
    用于计算以整张图为目标时的过杀率和漏杀率.

    对整体而言，
        * 过杀率定义为当GT为OK，但检测出了任意异常，则判定为过杀；
        * 漏杀率定义为当GT为异常数据，但没有检测出任意以上，则判定为漏杀。

    对每个类别而言，
        * 过杀率定义为当GT为任意缺陷（或则OK），但检测出其他缺陷，则判定为过杀；
        * 漏杀率定义为当GT为任意缺陷（或则OK），但没有检测出该缺陷，则判定为漏杀。
    """

    TARGET_OK = 'target_OK'
    TARGET_NG = 'target_NG'
    PREDICT_MISS = 'predict_miss'
    PREDICT_KILL = 'predict_kill'
    DEFAULT_CONFIG = EvaluatorDefaults.KillRateRelatedEvaluator_cfg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = merge_dict(
            EvaluatorDefaults.KillRateRelatedEvaluator_cfg, self.kwargs)
        self.score_threshold = self.kwargs.get('score_threshold')
        if self.score_threshold is None:
            self.score_threshold = [0] * self.num_classes
        self.OK_label = self.kwargs.get('OK_label')
        self.target_OK = np.array([0] * self.num_classes)
        self.target_NG = np.array([0] * self.num_classes)
        self.predict_miss = np.array([0] * self.num_classes)
        self.predict_kill = np.array([0] * self.num_classes)
        self.cls = np.array(range(self.num_classes)).reshape(1, self.num_classes, 1)
        self.epsilon = self.kwargs.get('epsilon')

    def single_process(self, inputs: Dict, outputs: Dict, **kwargs):
        pass

    def process(self, inputs_list: List[Dict], outputs_list: List[Dict],
                **kwargs):
        w_list = [
            outputs_list[idy][SegmentationModelOutputConstants.PREDICT_SCORES].shape[-1]
            for idy in range(len(outputs_list))
        ]
        h_list = [
            outputs_list[idy][SegmentationModelOutputConstants.PREDICT_SCORES].shape[-2]
            for idy in range(len(outputs_list))
        ]

        def accumulate(batch_predict, batch_target):
            batch = batch_predict.shape[0]

            predict_score = batch_predict.reshape(batch, self.num_classes, -1)  # N x C x HW
            predict_cls = predict_score.argmax(axis=1).astype(np.int)  # N x HW
            batch_target = batch_target.astype(np.int).reshape(batch, -1)  # N x HW
            batch_target_valid_num = (batch_target != self.ignore_label).sum(1)
            predict_cls[batch_target == self.ignore_label] = self.ignore_label

            score_valid = (predict_score >
                           np.asarray(self.score_threshold).reshape(1, self.num_classes, 1))  # N x C x HW
            predict_cls = np.repeat(predict_cls.reshape(batch, 1, -1), self.num_classes, axis=1) == self.cls
            predict_cls &= score_valid
            target = np.repeat(batch_target.reshape(batch, 1, -1), self.num_classes, axis=1) == self.cls  # N x C x HW
            target_sum_by_category = target.sum(2)
            target_if_this_label = np.ones_like(target_sum_by_category, dtype=np.bool)
            target_OK = np.ones(self.num_classes, dtype=np.int)
            for i in range(self.num_classes):
                if i == self.OK_label:
                    target_if_this_label[:, i] = (target_sum_by_category[:, i] == batch_target_valid_num[:])
                    target_OK[i] = target_if_this_label[:, i].sum()
                else:
                    target_if_this_label[:, i] = (target_sum_by_category[:, i] > 0)
                    target_OK[i] = (~target_if_this_label[:, i]).sum()
            target_NG = batch - target_OK
            self.target_OK += target_OK
            self.target_NG += target_NG

            predict_sum_by_category = predict_cls.sum(2)
            predict_if_this_label = np.ones_like(predict_sum_by_category, dtype=np.bool)
            for i in range(self.num_classes):
                if i == self.OK_label:
                    predict_if_this_label[:, i] = (predict_sum_by_category[:, i] == batch_target_valid_num[:])
                else:
                    predict_if_this_label[:, i] = (predict_sum_by_category[:, i] != 0)

            for i in range(self.num_classes):
                if i == self.OK_label:
                    self.predict_miss[i] += (predict_if_this_label & ~target_if_this_label)[:, i].sum(0)
                    self.predict_kill[i] += (~predict_if_this_label & target_if_this_label)[:, i].sum(0)
                else:
                    self.predict_miss[i] += (~predict_if_this_label & target_if_this_label)[:, i].sum(0)
                    self.predict_kill[i] += (predict_if_this_label & ~target_if_this_label)[:, i].sum(0)

        if min(w_list) != max(w_list) or min(h_list) != max(h_list):
            for idy in range(len(inputs_list)):
                predict = outputs_list[idy][SegmentationModelOutputConstants.PREDICT_SCORES][None, :]
                target = inputs_list[idy][SegmentationInputsConstants.TARGET][None, :]
                if SegmentationInputsConstants.WEIGHTS in inputs_list[idy]:
                    if inputs_list[idy][SegmentationInputsConstants.WEIGHTS] == 0:
                        continue
                    else:
                        accumulate(predict, target)
                else:
                    accumulate(predict, target)
        else:
            # TODO(zxyan): evaluate may slow.
            predicts = np.concatenate([
                outputs_list[idy][SegmentationModelOutputConstants.PREDICT_SCORES][None, :]
                for idy in range(len(outputs_list))
            ])
            targets = np.concatenate([
                inputs_list[idy][SegmentationInputsConstants.TARGET][None, :]
                for idy in range(len(inputs_list))
            ])
            weights = []
            for idy in range(len(inputs_list)):
                if SegmentationInputsConstants.WEIGHTS in inputs_list[idy]:
                    if inputs_list[idy][SegmentationInputsConstants.WEIGHTS] == 0:
                        weights.append(0)
                        continue
                if SegmentationInputsConstants.HEAD_WEIGHTS in inputs_list[idy]:
                    if inputs_list[idy][SegmentationInputsConstants.HEAD_WEIGHTS][self.head_id] == 0:
                        weights.append(0)
                        continue
                weights.append(1)
            weights = np.array(weights)
            predicts = predicts[weights == 1, ...]
            targets = targets[weights == 1, ...]
            if targets.shape[0] != 0 and predicts.shape[0] != 0:
                accumulate(predicts, targets)

    def reset(self):
        self.target_OK = np.array([0] * self.num_classes)
        self.target_NG = np.array([0] * self.num_classes)
        self.predict_miss = np.array([0] * self.num_classes)
        self.predict_kill = np.array([0] * self.num_classes)

    def calculate_kill_rate(self, inputs):
        results = {
            'whole':
                round(
                    (inputs[KillRateRelatedEvaluator.PREDICT_KILL][self.OK_label])
                    / (inputs[KillRateRelatedEvaluator.TARGET_OK][self.OK_label]), 4) * 100
        }
        for i in range(self.num_classes):
            if i != self.OK_label:
                results[self.label_map[i]] = \
                    round((inputs[KillRateRelatedEvaluator.PREDICT_KILL][i]) / (
                            inputs[KillRateRelatedEvaluator.TARGET_OK][i]), 4) * 100
        return results

    def calculate_miss_kill_rate(self, inputs):
        results = {
            'whole':
                round(
                    (inputs[KillRateRelatedEvaluator.PREDICT_MISS][self.OK_label])
                    / (inputs[KillRateRelatedEvaluator.TARGET_NG][self.OK_label]), 4) * 100
        }
        for i in range(self.num_classes):
            if i != self.OK_label:
                results[self.label_map[i]] = \
                    round((inputs[KillRateRelatedEvaluator.PREDICT_MISS][i]) / (
                            inputs[KillRateRelatedEvaluator.TARGET_NG][i]), 4) * 100
        return results

    def evaluate(self):
        if distributed():
            synchronize()

            target_OK_list = all_gather(self.target_OK)
            target_NG_list = all_gather(self.target_NG)
            predict_miss_list = all_gather(self.predict_miss)
            predict_kill_list = all_gather(self.predict_kill)

            if not is_main_process():
                return

            self.target_OK = np.array([0] * self.num_classes)
            self.target_NG = np.array([0] * self.num_classes)
            self.predict_miss = np.array([0] * self.num_classes)
            self.predict_kill = np.array([0] * self.num_classes)
            for each in target_OK_list:
                self.target_OK += each
            for each in target_NG_list:
                self.target_NG += each
            for each in predict_miss_list:
                self.predict_miss += each
            for each in predict_kill_list:
                self.predict_kill += each

        result = {}
        inputs = {
            KillRateRelatedEvaluator.TARGET_OK: self.target_OK,
            KillRateRelatedEvaluator.TARGET_NG: self.target_NG,
            KillRateRelatedEvaluator.PREDICT_MISS: self.predict_miss,
            KillRateRelatedEvaluator.PREDICT_KILL: self.predict_kill
        }
        result['kill_rate'] = self.calculate_kill_rate(inputs)
        result['miss_kill_rate'] = self.calculate_miss_kill_rate(inputs)
        result['score_threshold'] = {}
        for i in range(self.num_classes):
            result['score_threshold'][
                self.label_map[i]] = self.score_threshold[i]
        results = OrderedDict(result)
        return results


@EVALUATORS.register_module()
class ProductEvaluator(SegmentationEvaluatorBase):
    HIT = 'hit'
    MISS = 'miss'
    FA = 'fa'
    PREDICT = 'predict'
    TARGET = 'target'
    DEFAULT_CONFIG = EvaluatorDefaults.ProductEvaluator_cfg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = merge_dict(EvaluatorDefaults.ProductEvaluator_cfg, self.kwargs)
        self.ignore_BG = self.kwargs.get('ignore_BG')
        self.epsilon = self.kwargs.get('epsilon')
        self.predict = np.array([0] * self.num_classes)
        self.target = np.array([0] * self.num_classes)
        self.hit = np.array([0] * self.num_classes)
        self.miss = np.array([0] * self.num_classes)
        self.fa = np.array([0] * self.num_classes)

    def single_process(self, input: Dict, output: Dict, **kwargs):
        if isinstance(output[SegmentationModelOutputConstants.PREDICT_SCORES], np.ndarray):
            predict = output[SegmentationModelOutputConstants.PREDICT_SCORES].argmax(axis=0).astype(np.int)
            target = input[SegmentationInputsConstants.TARGET].astype(np.int)
        else:
            raise NotImplementedError('input support np.array only.')

        predict[target == self.ignore_label] = self.ignore_label
        k = (target >= 0) & (target < self.num_classes)

        hist_info = np.bincount(self.num_classes * target[k] + predict[k],
                                minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes)

        predict_s = hist_info.sum(0)
        target_s = hist_info.sum(1)
        hit = np.diag(hist_info)
        miss = target_s - hit
        fa = predict_s - hit
        inputs = {
            PixelBasedEvaluator.HIT: hit,
            PixelBasedEvaluator.MISS: miss,
            PixelBasedEvaluator.FA: fa,
            PixelBasedEvaluator.PREDICT: predict_s,
            PixelBasedEvaluator.TARGET: target_s
        }
        overall_iou = self.calculate_iou(inputs)
        return overall_iou

    def process(self, inputs_list: List[Dict], outputs_list: List[Dict],
                **kwargs):
        w_list = [
            outputs_list[idy][
                SegmentationModelOutputConstants.PREDICT_SCORES].shape[-1]
            for idy in range(len(outputs_list))
        ]
        h_list = [
            outputs_list[idy][
                SegmentationModelOutputConstants.PREDICT_SCORES].shape[-2]
            for idy in range(len(outputs_list))
        ]

        def accumulate(batch_predict, batch_target):
            batch_predict = batch_predict.argmax(axis=1).astype(np.int)
            batch_target = batch_target.astype(np.int)

            batch_predict[batch_target ==
                          self.ignore_label] = self.ignore_label
            k = (batch_target >= 0) & (batch_target < self.num_classes)

            hist_info = np.bincount(
                self.num_classes * batch_target[k] + batch_predict[k],
                minlength=self.num_classes ** 2).reshape(self.num_classes,
                                                         self.num_classes)

            predict_s = hist_info.sum(0)
            target_s = hist_info.sum(1)
            hit = np.diag(hist_info)
            miss = target_s - hit
            fa = predict_s - hit

            self.predict += predict_s
            self.target += target_s
            self.hit += hit
            self.miss += miss
            self.fa += fa

        if min(w_list) != max(w_list) or min(h_list) != max(h_list):
            for idy in range(len(inputs_list)):
                predict = outputs_list[idy][SegmentationModelOutputConstants.PREDICT_SCORES][None, :]
                target = inputs_list[idy][SegmentationInputsConstants.TARGET][None, :]
                accumulate(predict, target)
        else:
            for idy in range(len(outputs_list)):
                predict = np.concatenate([
                    outputs_list[idy][SegmentationModelOutputConstants.PREDICT_SCORES][None, :]
                ])
                target = np.concatenate([
                    inputs_list[idy][SegmentationInputsConstants.TARGET][None, :]
                ])
                accumulate(predict, target)

    def reset(self):
        self.predict = np.array([0] * self.num_classes)
        self.target = np.array([0] * self.num_classes)
        self.hit = np.array([0] * self.num_classes)
        self.miss = np.array([0] * self.num_classes)
        self.fa = np.array([0] * self.num_classes)

    def calculate_iou(self, inputs):
        results = {}
        iou = inputs[ProductEvaluator.HIT] / (
                inputs[ProductEvaluator.HIT] + inputs[ProductEvaluator.MISS] +
                inputs[ProductEvaluator.FA] + self.epsilon)

        for i in range(1 if self.ignore_BG else 0, self.num_classes):
            results[self.label_map[i]] = iou[i]

        results['mean'] = np.nanmean(iou if not self.ignore_BG else iou[1:])
        return results

    def calculate_precision(self, inputs):
        results = {}

        precision = inputs[ProductEvaluator.HIT] / (
                inputs[ProductEvaluator.PREDICT] + self.epsilon)
        precision[precision > 100] = 0.

        for i in range(1 if self.ignore_BG else 0, self.num_classes):
            results[self.label_map[i]] = precision[i]

        results['mean'] = np.nanmean(
            precision if not self.ignore_BG else precision[1:])
        return results

    def calculate_recall(self, inputs):
        results = {}

        recall = inputs[ProductEvaluator.HIT] / (
                inputs[ProductEvaluator.TARGET] + self.epsilon)
        recall[recall > 100] = 0.

        for i in range(1 if self.ignore_BG else 0, self.num_classes):
            results[self.label_map[i]] = recall[i]

        results['mean'] = np.nanmean(
            recall if not self.ignore_BG else recall[1:])
        return results

    def evaluate(self):
        """
        the format of result.json
        {
            "overall_index":
                {
                    "recall": 0.2551890799028579,
                    "map": 0.18356431979828658,
                    "precision": 0.18356431979828658
                },
            "classes_index": [
                {
                    "Index": 1,
                    "Type": "1",
                    "Recall": 0.2551890799028579,
                    "Precision": 0.18356431979828658,
                    "IOU": 0.11952651117181477
                }
            ]
        }
        :return:
        """
        if distributed():
            synchronize()

            predict_list = all_gather(self.predict)
            target_list = all_gather(self.target)
            hit_list = all_gather(self.hit)
            miss_list = all_gather(self.miss)
            fa_list = all_gather(self.fa)
            if not is_main_process():
                return

            self.predict = np.array([0] * self.num_classes)
            self.target = np.array([0] * self.num_classes)
            self.hit = np.array([0] * self.num_classes)
            self.miss = np.array([0] * self.num_classes)
            self.fa = np.array([0] * self.num_classes)
            for each in predict_list:
                self.predict += each
            for each in target_list:
                self.target += each
            for each in hit_list:
                self.hit += each
            for each in miss_list:
                self.miss += each
            for each in fa_list:
                self.fa += each

        inputs = {
            PixelBasedEvaluator.HIT: self.hit,
            PixelBasedEvaluator.MISS: self.miss,
            PixelBasedEvaluator.FA: self.fa,
            PixelBasedEvaluator.PREDICT: self.predict,
            PixelBasedEvaluator.TARGET: self.target
        }
        overall_iou = self.calculate_iou(inputs)
        overall_precision = self.calculate_precision(inputs)
        overall_recall = self.calculate_recall(inputs)
        result_json = {
            'overall_index': {
                'recall': overall_recall['mean'],
                'map': overall_precision['mean'],
                'precision': overall_precision['mean'],
                'iou': overall_iou['mean'],
            },
            'classes': []
        }
        for i in range(1 if self.ignore_BG else 0, self.num_classes):
            result_json['classes'].append({
                'class': self.label_map[i],
                'Recall': overall_recall[self.label_map[i]],
                'Precision': overall_precision[self.label_map[i]],
                'IOU': overall_iou[self.label_map[i]]
            })
        result_json = OrderedDict(result_json)
        return result_json
