function Recon_k = make_Recon_k(True_k, pat, nacs, kSize, lamda)
    [feRes, peRes, zRes, cRes, dRes, eRes] = size(True_k);

    measc_cc_us = True_k.*pat;
    measc_cc_us_vcc = jh_VCC_util(measc_cc_us);
    ACS = True_k(:,floor(peRes/2)+1+ceil(-nacs/2):floor(peRes/2)+ceil(nacs/2),:,:,:,:);
    ACS_vcc = jh_VCC_util(ACS);
    
    Recon_k = zeros([feRes, peRes, zRes, cRes, dRes, eRes]);
    
    nind = zRes;
    upd = textprogressbar(nind, 'barlength', 20, ...
                             'updatestep', 1, ...
                             'startmsg', 'JVC_GRAPPA',...
                             'endmsg', ' Done.', ...
                             'showbar', false, ...
                             'showremtime', true, ...
                             'showactualnum', false, ...
                             'barsymbol', '+', ...
                             'emptybarsymbol', '-');
    ind = 1;
    
    for zz = 1:zRes
        calibKernel = squeeze(ACS_vcc(:,:,zz,:,:,:));
        tmp = im2row(calibKernel, kSize);
        [winSize, winCount,tN] = size(tmp);
        A = reshape(tmp, [winSize, winCount*tN]);
        AtA = A'*A;
        clear tmp A;

        DATA_us = squeeze(measc_cc_us_vcc(:,:,zz,:,:,:));
        Recon_k(:,:,zz,:,:) = jh_ARC(DATA_us, AtA, kSize, lamda);
        upd(ind);
        ind=ind+1;
    end 

    Recon_k(:,floor(peRes/2)+1+ceil(-nacs/2):floor(peRes/2)+ceil(nacs/2),:,:,:,:) = ACS(:,:,:,:,:,:);

    Recon_k = single(Recon_k);
end