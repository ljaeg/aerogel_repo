function stacked_img = stack_all(path_)

big_box = uint8(zeros(60,384,512,3));

jj = 1;
while true
	try
		big_box(jj, :, :, :) = imread([path_ num2str(jj) '.png']);
		jj = jj + 1;
	catch
		break
	end
end

big_box = big_box(1:jj-1, :, :, :);
cstack1 = cell(jj-1, 1);
cstack2 = cell(384, 1);
cstack3 = cell(512, 1);

for (ii=1:size(big_box, 1))
	cstack1{ii} = squeeze(big_box(ii, :, :, :));
end

for (ii=1:384)
	cstack2{ii} = squeeze(big_box(:, ii, :, :));
end

for (ii=1:512)
	cstack3{ii} = squeeze(big_box(:, :, ii, :));
end

stacked_img = fstack(cstack1);
x = fstack(cstack2);
y = fstack(cstack3);
%image(stacked_img)
%pause(10)