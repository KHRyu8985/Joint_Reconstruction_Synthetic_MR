function mask = JLORAKS_mask_util_jh(num_acs, feRes, peRes, zRes, cRes, Cont, R_fe, R_pe, shift_fe, shift_pe)
    Mask_acs = zeros([feRes, peRes, zRes, cRes*Cont]);
    Mask_acs(1+end/2-num_acs(1)/2:end/2+num_acs(1)/2 + 1, 1+end/2-num_acs(2)/2:end/2+num_acs(2)/2 + 1, :, :) = 1;

    Mask_sampled = zeros([feRes, peRes, zRes, cRes*Cont]);

    for cc = 1:cRes*Cont
        Mask_sampled(1+shift_fe(cc):R_fe:end, 1+shift_pe(cc):R_pe:end, :, cc) = 1;
    end

    mask = reshape(Mask_acs + Mask_sampled, feRes, peRes, zRes, cRes, Cont);
    mask = mask>0;
end