import torch
import torch.nn as nn
import torch.nn.functional as F


class yololoss(nn.Module):
    def __init__(self, args, l_coord, l_noobj):
        super(yololoss, self).__init__()
        self.S = args.yolo_S # S is the grid size
        self.B = args.yolo_B # B is the number of bounding boxes
        self.C = args.yolo_C # C is the classification 
        self.len_pred = (5 * self.B) + self.C
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        """ compute IOU between boxes
            - box1 (bs, 4)  4: [x1, y1, x2, y2]  left top and right bottom # bs is the batch size
            - box2 (bs, 4)  4: [x1, y1, x2, y2]  left top and right bottom
        """
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max( # left top # print error should be left bottom
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min( # right bottom # print error should be right top
            box1[:, 2:4].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:4].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M] # width * height

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou # iou has shape (boxes, iou)

    def forward(self, prediction, target):
        """
            - prediction: (bs, S, S, B*5+C)    [x1, y1, w1, h1, c1, x2, y2, w2, h2, c2, confidence for C classes]
            - target: (bs, S, S, B*5+C)    [x, y, w, h, c, x, y, w, h, c, confidence for C classes]
        """        
        N = prediction.size()[0]               # coo_mask mask the GRID with target[:, :, :, 4] > 0, noo_mask mask the GRID with target[:, :, :, 4] = 0!!!!!!!!!
        coo_mask = target[:, :, :, 4] > 0       # 4 is the 5 fifth place which is the confidence, mask means using boolean to extract from matrix by condition
        noo_mask = target[:, :, :, 4] == 0        # confidence=0 means no object
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target)      # unsqueeze to [:, :, :, -1], expand to [:, :, :, :]
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target)

        coo_pred = prediction[coo_mask].view(-1, self.len_pred)       # convert to 2D form, contain only the GRID where target[:, :, :, 4] > 0 !!!!!!!!!, each column is the element of len_pred, each row is a grid, (grids, len_pred)
        box_pred = coo_pred[:, :self.B * 5].contiguous().view(-1, 5)     # boxes [[x1,y1,w1,h1,c1]; [x2,y2,w2,h2,c2]]
        class_pred = coo_pred[:, self.B * 5:]

        coo_target = target[coo_mask].view(-1, self.len_pred)            # convert to 2D form, contain only the GRID where target[:, :, :, 4] > 0
        box_target = coo_target[:, :self.B * 5].contiguous().view(-1, 5)       # shape (grids, 5) one row of len_pred becomes two row of self.B if B=2, [[grid1, box1], [grid1, box2]], it is for one grid
        class_target = coo_target[:, self.B * 5:]

        """Non Maximum Suppression"""
        coo_response_mask = torch.cuda.BoolTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.BoolTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()
        for i in range(0, box_target.size()[0], self.B):       # for each grid
            box1 = box_pred[i:i + self.B]                     # contains info of predicted boxes for every grid, for b = 2, [[grid1, box1], [grid1, box2]]
            box1_xyxy = torch.FloatTensor(box1.size())
            """ from [x,y,w,h] to [x1,y1,x2,y2]"""
            box1_xyxy[:, :2] = box1[:, :2] / self.S - 0.5 * box1[:, 2:4]    # calculate left bottom, box1[:, :2] is the coord of centre, box1[:, 2:4] is the width and height
            box1_xyxy[:, 2:4] = box1[:, :2] / self.S + 0.5 * box1[:, 2:4]    # calculate right top

            box2 = box_target[i].view(-1, 5)           # the first ground truth box for every grid
            box2_xyxy = torch.FloatTensor(box2.size())
            ###################################################################
            # TODO: Please fill the codes below to calculate the iou of the two boxes and substite the "?"
            # Note: return variable: iou_res (self.B, 1)
            ##################################################################
            
            box2_xyxy[:, :2] = box2[:, :2] / self.S - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / self.S + 0.5 * box2[:, 2:4]
            pass

            iou_res = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])      # computation of maximum iou for predicted boxes to the first ground truth box of the same grid
            ##################################################################
            max_iou, max_index = iou_res.max(0)        # max(0) is N (index for box in prediction) index is the box of maximum iou in the iou matrix, there are grids(only confidence>0) number of matrix
            max_index = max_index.data.cuda()

            coo_response_mask[i + max_index] = 1       # mask for the max iou box in prediction to the first ground truth box in target, has shape (grids, 5), 1 means true
            
            for j in range(self.B):         # not response contain the boxes that does not show the largest condidence
                if j == max_index:
                    continue
                else:
                    coo_not_response_mask[i + j] = 1
            # confidence score = IOU of predicted box and the ground truth 
            box_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = max_iou.data.cuda()       # in target substitute the fifith place confidence to the max iou at the index of box of max iou in prediction
        
        box_target_iou = box_target_iou.cuda()
        
        """Compute Term1 + Term2: Location Loss"""
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)

        ###################################################################
        # TODO: Please fill the codes below to calculate the location loss
        ##################################################################
        
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum', size_average=False) + \
                   F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), reduction='sum', size_average=False)
        pass
        

        ##################################################################
        
        """Compute the 3rd Term: IOU loss for boxes containing the objects"""
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], reduction='sum', size_average=False)
        
        """Compute the 4th Term (Part I): Not Response Loss"""
        ###################################################################
        # TODO: Please fill the codes below to calculate the Not Response Loss
        ##################################################################
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response_iou = box_target_iou[coo_not_response_mask].view(-1, 5)
        box_target_not_response_iou[:, 4] = 0
        not_response_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response_iou[:, 4], reduction='sum', size_average=False)
        pass


        ##################################################################
        
        """Compute the 4th Term (Part II): No Object Contain Loss"""
        noo_pred = prediction[noo_mask].view(-1, self.len_pred)
        noo_target = target[noo_mask].view(-1, self.len_pred)
        noo_pred_mask = torch.cuda.BoolTensor(noo_pred.size())
        noo_pred_mask.zero_()
        for i in range(self.B):
            noo_pred_mask[:, i*5+4] = 1
        noo_pred_c = noo_pred[noo_pred_mask]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, reduction='sum', size_average=False)
        
        """Compute the 5th Term: Class Loss"""
        class_loss = F.mse_loss(class_pred, class_target, reduction='sum', size_average=False)
        
        """Summarize the five terms"""
        loss = self.l_coord * loc_loss + contain_loss + self.l_noobj * (not_response_loss +  nooobj_loss) + class_loss

        return loss / N