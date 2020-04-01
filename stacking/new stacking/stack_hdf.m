function stack_hdf(df_path, save_path)

ttv = {'Train', 'Test', 'Val'};
yn = {'Yes', 'No'};
tic
for ii=1:length(ttv)
	for jj=1:length(yn)
		ds = h5read(df_path, strcat('/', ttv{ii}, yn{jj}));
		stacked_Z = cell(size(ds, 5), 1);
		stacked_X = cell(size(ds, 5), 1);
		stacked_Y = cell(size(ds, 5), 1);
		for mm=1:size(ds, 5) %I believe it's 5 but double check
			Z, X, Y = stack_single_movie(ds(:, :, :, :, mm));
			stacked_Z{mm} = Z;
			stacked_X{mm} = X;
			stacked_Y{mm} = Y;
		%then put the stacked stuff into an hdf5 file, change their classes if need be
		stacked_Z = cell2mat(stacked_Z);
		stacked_X = cell2mat(stacked_X);
		stacked_Y = cell2mat(stacked_Y);

		%Z
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
