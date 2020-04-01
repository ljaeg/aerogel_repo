function mlsandbox()

stack = uint8(zeros(384,512,3,45));
cstack = cell(45, 1)

for(i = 1:45)
    if (i < 10)
        bbb = ['00' num2str(i)];
    elseif (i < 100)
        bbb = ['0' num2str(i)];
    end;
 figure(1);
  
 a = imread(['http://s3.amazonaws.com/stardustathome.testbucket/real/fm_-60712_-54519/fm_-60712_-54519-' bbb '.jpg']);
pause(0.01);
image(a);

cstack{i} = a;
%cstack = [cstack {a}];

end;

stackp = permute(stack,[1 2 4 3]);


cstack = {};
figure(1);
for(i = 1:45)
    a = squeeze(stackp(:,:,i,:));
    cstack = [cstack {a}];

    image(a);
    title(num2str(i));
    pause(0.1);
end;


zflattened = fstack(cstack);
figure(11);
image(zflattened);



% x
cstack = {};
figure(2);
for(i = 1:512)
    a = squeeze(stackp(:,i,:,:));
    a = permute(a,[2 1 3]);
    cstack = [cstack {a}];
    image(a);
    title(num2str(i));
    pause(0.01);
end;


%  'logsize'    : size of the LoG (laplacian of gaussian) filter used for 
%                 detecting pixel in focus, default is 13
%
%  'logstd'     : standard deviation for LoG filter, default is 2
%  'dilatesize' : size of structure element used to smooth the focus
%                 detection result, default is 31
%  'blendsize'  : size of the Guassian filter for bleding pixels taken from
%                 different focal planes, default is 31
%  'blendstd'   : standard deviation of the Gaussian filter for blending
%                 pixels from different planes, default is 5
%  'logthreshold' : threshold for logresponse, default is 0
%  

xflattened = fstack(cstack, 'blendsize',31);
figure(21);
image(xflattened);


% y
cstack = {};
figure(3);
for(i = 1:384)
    a = squeeze(stackp(i,:,:,:));
    a = permute(a,[2 1 3]);
    cstack = [cstack {a}];
    image(a);
    title(num2str(i));
    pause(0.01);
end;

yflattened = fstack(cstack);
figure(31);
image(yflattened);


