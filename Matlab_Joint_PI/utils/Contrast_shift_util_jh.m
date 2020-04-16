function [shift_fe, shift_pe] = Contrast_shift_util_jh(del_fe, del_pe, cRes)
    shift_fe = repmat(del_fe, [cRes,1]);
    shift_fe = shift_fe(:).';
    shift_fe = repmat(shift_fe,1,2);

    shift_pe = repmat(del_pe, [cRes,1]);
    shift_pe = shift_pe(:).';
    shift_pe = repmat(shift_pe,1,2);
end
