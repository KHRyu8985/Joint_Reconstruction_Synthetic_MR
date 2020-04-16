function JLORAKS_k = JLORAKS_util_jh(partial_under_k, tmp_mask, neighbor_R, r_S, pcg_tol, pcg_iter)
    N1 = size(partial_under_k,1);
    N2 = size(partial_under_k,2);
    Nc = size(partial_under_k,3);
    
    % Find loraks annhilating filters 
    Nic = calib_loraks( partial_under_k, tmp_mask, neighbor_R, r_S );

    S.type = '()';
    S.subs{:} = find(~vect(ifftshift(tmp_mask)));

    phi = @(x) subsref(vect(fft2(reshape(sum(reshape(bsxfun(@times,Nic,vect(ifft2(subsasgn(zeros(N1,N2,Nc),S,x)))),[N1,N2,Nc,Nc]),3),[N1 N2 Nc]))),S);
    b = -subsref(vect(fft2(reshape(sum(reshape(bsxfun(@times,Nic,vect(ifft2(reshape(ifftshift(partial_under_k),[N1,N2,Nc])))),[N1,N2,Nc,Nc]),3),[N1 N2 Nc]))),S);

    % PCG recon
    tic
        [z, flag, res, iter, resvec] = pcg(@(x)  phi(x), b, pcg_tol, pcg_iter);
    toc
    
    A = @(x) fftshift(ifftshift(partial_under_k) + subsasgn(zeros(N1,N2,Nc),S,x));
    JLORAKS_k = A(z);
end