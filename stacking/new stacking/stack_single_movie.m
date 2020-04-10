function [Z, X, Y] = stack_single_movie(big_box)

[z, x, y, c] = size(big_box);
cstack1 = cell(z, 1);
cstack2 = cell(x, 1);
cstack3 = cell(y, 1);

for (ii=1:z)
	cstack1{ii} = squeeze(big_box(ii, :, :, :));
end

for (ii=1:x)
	cstack2{ii} = squeeze(big_box(:, ii, :, :));
end

for (ii=1:y)
	cstack3{ii} = squeeze(big_box(:, :, ii, :));
end

Z = fstack(cstack1);
X = fstack(cstack2);
Y = fstack(cstack3);