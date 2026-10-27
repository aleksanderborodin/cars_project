//! Stable C ABI v1. Caller owns buffers; handles are created/freed here.
//! Valid handles must be used serially. Null/short inputs produce safe no-ops.
//! Raw pointer lengths must describe valid allocated buffers; Rust panics are
//! contained at the ABI boundary, but this cannot sandbox unsafe candidates.
mod policy;
use policy::Controller;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[no_mangle]
pub extern "C" fn carctl_abi_version() -> u32 { 1 }

#[no_mangle]
pub extern "C" fn carctl_new() -> *mut c_void {
    Box::into_raw(Box::new(Controller::default())).cast()
}

#[no_mangle]
/// # Safety
/// p is null or a live handle returned by carctl_new, freed exactly once.
pub unsafe extern "C" fn carctl_free(p: *mut c_void) {
    if !p.is_null() { drop(Box::from_raw(p.cast::<Controller>())); }
}

#[no_mangle]
/// # Safety
/// track has n_tiles * stride floats; p is a live exclusive handle.
pub unsafe extern "C" fn carctl_reset(p: *mut c_void, track: *const f32, n_tiles: u32, stride: u32) {
    if p.is_null() || track.is_null() || stride < 4 || n_tiles == 0 { return; }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let data = std::slice::from_raw_parts(track, n_tiles as usize * stride as usize);
        let tiles = data.chunks_exact(stride as usize).map(|t| [t[0], t[1], t[2], t[3]]).collect();
        (*p.cast::<Controller>()).reset(tiles);
    }));
}

#[no_mangle]
/// # Safety
/// state has n floats; out has 3 writable floats; p is a live exclusive handle.
pub unsafe extern "C" fn carctl_act(p: *mut c_void, state: *const f32, n: u32, out: *mut f32) {
    if out.is_null() { return; }
    let result = if p.is_null() || state.is_null() || n < 20 {
        [0.0; 3]
    } else {
        catch_unwind(AssertUnwindSafe(|| {
            (*p.cast::<Controller>()).act(std::slice::from_raw_parts(state, n as usize))
        })).unwrap_or([0.0; 3])
    };
    std::ptr::copy_nonoverlapping(result.as_ptr(), out, 3);
}
