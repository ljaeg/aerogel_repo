function stack_hdf(df_path, save_path)

ttv = {'Train', 'Test', 'Val'};
yn = {'Yes', 'No'};

for ii=1:length(ttv)
	for jj=1:length(yn)
		ds = h5read(df_path, strcat('/', ttv{ii}, yn{jj}));
		stacked_Z = class(size(ds, 5), 1);
		stacked_X = class(size(ds, 5), 1);
		stacked_Y = class(size(ds, 5), 1);
		for mm=1:size(ds, 5) %I believe it's 5 but double check
			Z, X, Y = stack_single_movie(ds(:, :, :, :, mm));
			stacked_Z{mm} = Z;
			stacked_X{mm} = X;
			stacked_Y{mm} = Y;
		%then put the stacked stuff into an hdf5 file, change their classes if need be
	end
end
