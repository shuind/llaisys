#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    if (ndim() == 0 || numel() == 0 ) return true;
    size_t expected = 1;
    for(size_t d = ndim() ; d-- > 0; ){
        if (shape()[d] != 1 && strides()[d] != expected){
            return false;
        }
        expected *= shape()[d];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t n = ndim();
    CHECK_ARGUMENT(order.size() == n, "permute: order.size() != ndim");
    std::vector<bool> vis(n, false);
    for (size_t i = 0; i < n; ++i) {
        CHECK_ARGUMENT(order[i] < n, "permute: order index out of range");
        CHECK_ARGUMENT(!vis[order[i]], "permute: order has duplicate indices");
        vis[order[i]] = true;
    }
    TensorMeta new_meta = _meta;
    new_meta.shape.resize(n);
    new_meta.strides.resize(n);
    for (size_t i = 0; i < n; ++i) {
        new_meta.shape[i] = shape()[order[i]];
        new_meta.strides[i] = strides()[order[i]];
    }
    return std::shared_ptr<Tensor>(new Tensor(std::move(new_meta), _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t new_numel = 1;
    for (size_t s : shape) new_numel *= s;
    CHECK_ARGUMENT(new_numel == numel(), "view: new_numel != numel()");

    if (numel() == 0) {
        std::vector<ptrdiff_t> new_strides(shape.size());
        size_t st = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            new_strides[i] = (ptrdiff_t)st;
            st *= shape[i];
        }
        TensorMeta new_meta{dtype(), shape, new_strides};
        return std::shared_ptr<Tensor>(new Tensor(std::move(new_meta), _storage, _offset));
    }

    std::vector<size_t> old_shape_n1;
    std::vector<ptrdiff_t> old_stride_n1;
    old_shape_n1.reserve(ndim());
    old_stride_n1.reserve(ndim());
    for (size_t i = 0; i < ndim(); ++i) {
        if (_meta.shape[i] != 1) {
            old_shape_n1.push_back(_meta.shape[i]);
            old_stride_n1.push_back(_meta.strides[i]);
        }
    }

    std::vector<size_t> new_shape_n1;
    new_shape_n1.reserve(shape.size());
    for (size_t s : shape) {
        if (s != 1) new_shape_n1.push_back(s);
    }

    if (old_shape_n1.empty()) {
        std::vector<ptrdiff_t> out_strides(shape.size());
        for (size_t i = shape.size(); i-- > 0;) {
            if (i == shape.size() - 1) out_strides[i] = 1;
            else out_strides[i] = out_strides[i + 1];
        }
        TensorMeta new_meta{dtype(), shape, out_strides};
        return std::shared_ptr<Tensor>(new Tensor(std::move(new_meta), _storage, _offset));
    }

    std::vector<ptrdiff_t> new_stride_n1(new_shape_n1.size(), 0);

    ptrdiff_t vd = (ptrdiff_t)new_shape_n1.size() - 1; 
    ptrdiff_t td = (ptrdiff_t)old_shape_n1.size() - 1; 

    while (td >= 0) {
        ptrdiff_t chunk_base_stride = old_stride_n1[(size_t)td];
        size_t chunk_numel = old_shape_n1[(size_t)td];

        while (td > 0) {
            size_t inner_size = old_shape_n1[(size_t)td];
            ptrdiff_t inner_stride = old_stride_n1[(size_t)td];
            ptrdiff_t outer_stride = old_stride_n1[(size_t)td - 1];

            if (outer_stride == inner_stride * (ptrdiff_t)inner_size) {
                td--;
                chunk_numel *= old_shape_n1[(size_t)td];
                continue;
            }
            break;
        }

        size_t packed = 1;
        size_t cur_stride = (size_t)chunk_base_stride;

        while (vd >= 0 && packed < chunk_numel) {
            new_stride_n1[(size_t)vd] = (ptrdiff_t)cur_stride;
            packed *= new_shape_n1[(size_t)vd];
            cur_stride *= new_shape_n1[(size_t)vd];
            vd--;
        }

        CHECK_ARGUMENT(packed == chunk_numel, "view: incompatible reshape (chunk mismatch)");

        td--; 
    }

    CHECK_ARGUMENT(vd < 0, "view: incompatible reshape (extra dims)");

    std::vector<ptrdiff_t> out_strides(shape.size(), 0);
    ptrdiff_t j = (ptrdiff_t)new_stride_n1.size() - 1;

    for (ptrdiff_t i = (ptrdiff_t)shape.size() - 1; i >= 0; --i) {
        if (shape[(size_t)i] == 1) {
            if ((size_t)i == shape.size() - 1) out_strides[(size_t)i] = 1;
            else out_strides[(size_t)i] = out_strides[(size_t)i + 1];
        } else {
            out_strides[(size_t)i] = new_stride_n1[(size_t)j];
            j--;
        }
    }

    TensorMeta new_meta{dtype(), shape, out_strides};
    return std::shared_ptr<Tensor>(new Tensor(std::move(new_meta), _storage, _offset));
}


tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < ndim(), "slice : dim >= ndim");
    CHECK_ARGUMENT(start <= end, "slice : start > end");
    CHECK_ARGUMENT(end <= shape()[dim], "slice : end > shape()[dim]");
    TensorMeta new_meta = _meta;
    new_meta.shape[dim] = end - start;

    size_t delta_bytes = start * strides()[dim] * elementSize();
    size_t new_offset = _offset + delta_bytes;

    return std::shared_ptr<Tensor>(new Tensor(std::move(new_meta), _storage, new_offset));
}

void Tensor::load(const void *src_) {
    CHECK_ARGUMENT(src_ != nullptr, "load: src is null");
    CHECK_ARGUMENT(isContiguous(), "load: not contiguous ");

    size_t bytes = numel() * elementSize();

    if (deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(data(), src_, bytes);
        return;
    }

    core::context().setDevice(deviceType(), deviceId());
    core::context().runtime().api()->memcpy_sync(
        data(), src_, bytes, LLAISYS_MEMCPY_H2D);
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    size_t new_numel = 1;
    for (size_t s : shape) new_numel *= s;
    CHECK_ARGUMENT(new_numel == numel(), "reshape: new_numel != numel()");
    if (isContiguous()) {
        return view(shape);
    }
    auto tmp = contiguous();
    return tmp->view(shape);
}


tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
