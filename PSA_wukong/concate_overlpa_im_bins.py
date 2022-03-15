def concate_img(res_img, padded_ims, stride_h, stride_w, bin_size, bins):

    def fun_(res_img, bin_, y_, j):
        for i in range(bin_):
            if i == 0 and y_ == 0:
                res_img[:bin_size, y_:y_+bin_size, :] += padded_ims[i*bins[1]+j]
            elif y_ == 0:
                cur = padded_ims[i*bins[1]+j][-stride_h:, :, :]
                res_img[bin_size+(i-1)*stride_h:+bin_size+i*stride_h, y_:y_+bin_size, :] += cur
            else:
                cur = padded_ims[i*bins[1]+j][-stride_h:, -stride_w:, :]
                res_img[bin_size+(i-1)*stride_h:+bin_size+i*stride_h, y_:y_+stride_w, :] += cur
        if j == 0:
            y_ += bin_size
        else:
            y_ += stride_w*j

        return y_

    y_ = 0
    for j in range(bins[1]-1):
        y_  = fun_(res_img, bins[0], y_, j)