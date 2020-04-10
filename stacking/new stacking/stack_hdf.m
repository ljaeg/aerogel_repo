function stack_hdf(df_path, save_path)

ttv = {'Train', 'Test', 'Val'};
yn = {'Yes', 'No'};
tic
for ii=1:length(ttv)
	for jj=1:length(yn)
		ds = h5read(df_path, strcat('/', ttv{ii}, yn{jj}));
		ds = ds(1:3, :, :, :, :);
		ds = permute(ds, [5 4 2 3 1]);
		data_size = size(ds)
		stacked_Z = zeros(data_size([5 3 4 1])); %cell(size(ds, 1), 1);
		stacked_X = zeros(data_size([5 2 4 1])); %cell(size(ds, 1), 1);
		stacked_Y = zeros(data_size([5 2 3 1])); %cell(size(ds, 1), 1);
		for mm=1:size(ds, 1) 
			[Z, X, Y] = stack_single_movie(squeeze(ds(mm, :, :, :, :)));
			Z = permute(Z, [3 1 2]);
			X = permute(X, [3 1 2]);
			Y = permute(Y, [3 1 2]);
			stacked_Z(:, :, :, mm) = Z;
			stacked_X(:, :, :, mm) = X;
			stacked_Y(:, :, :, mm) = Y;
		end
		%then put the stacked stuff into an hdf5 file

		%Z
		size(stacked_Z)
		h5create(save_path, strcat('/', ttv{ii}, yn{jj}, '_Z'), size(stacked_Z));
		h5write(save_path, strcat('/', ttv{ii}, yn{jj}, '_Z'), stacked_Z);

		%X
		h5create(save_path, strcat('/', ttv{ii}, yn{jj}, '_X'), size(stacked_X));
		h5write(save_path, strcat('/', ttv{ii}, yn{jj}, '_X'), stacked_X);

		%Y
		h5create(save_path, strcat('/', ttv{ii}, yn{jj}, '_Y'), size(stacked_Y));
		h5write(save_path, strcat('/', ttv{ii}, yn{jj}, '_Y'), stacked_Y);

		toc
	end
end
