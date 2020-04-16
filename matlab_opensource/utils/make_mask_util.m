function mask = make_mask_util(True_k, pat, nacs)
    [feRes, peRes, zRes, cRes, dRes, eRes] = size(True_k);
    ACS = True_k(:,floor(peRes/2)+1+ceil(-nacs/2):floor(peRes/2)+ceil(nacs/2),:,:,:,:);
    mask = pat;
    mask(:,floor(peRes/2)+1+ceil(-nacs/2):floor(peRes/2)+ceil(nacs/2),:,:,:,:) = ACS(:,:,:,:,:,:);
    mask(abs(mask)>0) = 1;
    
    mask = single(mask);
end