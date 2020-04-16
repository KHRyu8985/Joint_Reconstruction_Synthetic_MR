function ACS_vcc = jh_VCC_util(k_spcae)

[feRes, peRes, zRes, cRes, dRes, eRes] = size(k_spcae);
    
VCC_signals = conj(flip(flip(k_spcae,1),2));
if mod(peRes,2) == 0
    VCC_signals = circshift(VCC_signals,[0 1 0 0]);
end

if mod(feRes,2) == 0
    VCC_signals = circshift(VCC_signals,[1 0 0 0]);
end
ACS_vcc = cat(4,k_spcae,VCC_signals);
end














