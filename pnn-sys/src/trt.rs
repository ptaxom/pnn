
extern "C" {
    pub fn builder_create(dtype: usize, batchsize: usize) -> *mut c_void;

    pub fn builder_destroy(builder: *mut c_void);

    pub fn builder_add_convolution(builder: *mut c_void,
        input_id: usize,
        feature_maps: usize,
        input_c: usize,
        kernel_size: usize,
        padding: usize,
        stride: usize,
        kernels: *const f32,
        biases: *const f32
    ) -> c_int;

    pub fn builder_add_activation(builder: *mut c_void, ind: usize, name: *const c_char) -> c_int;

    pub fn builder_add_shortcut(builder: *mut c_void, n_layers: usize, indeces: *const usize) -> c_int;

    pub fn builder_add_upsample(builder: *mut c_void, ind: usize, stride: usize) -> c_int;

    pub fn builder_add_input(builder: *mut c_void, name: *const c_char, channels: usize, width: usize, height: usize) -> c_int;

    pub fn builder_add_yolo(builder: *mut c_void, ind: usize, name: *const c_char);

    pub fn builder_add_route(builder: *mut c_void, n_layers: usize, indeces: *const usize ) -> c_int;

    pub fn builder_add_pooling(builder: *mut c_void,
        ind: usize,
        stride: usize,
        window_size: usize,
        padding: usize,
        is_max: usize
    ) -> c_int;

    pub fn builder_build(builder: *mut c_void, avg_iters: usize, min_iters: usize, engine_path: *const c_char) -> c_int;
}