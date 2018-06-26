extern crate failure;
extern crate gfx_hal as hal;
extern crate gfx_render as render;

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

use std::borrow::{Borrow, Cow};

use failure::Error;

use hal::format::{Aspects, Format, Swizzle};
use hal::image::{
    Access, Kind, Layout, Offset, StorageFlags, SubresourceLayers, SubresourceRange, Tiling, Usage,
    ViewKind,
};
use hal::memory::Properties;
use hal::queue::QueueFamilyId;
use hal::{Backend, Device};

use render::{Factory, Image};

/// Texture builder allow user to build texture
/// specifying image kind, format and data properties.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TextureBuilder<'a> {
    kind: Kind,
    format: Format,
    data_width: u32,
    data_height: u32,
    data: Cow<'a, [u8]>,
}

impl<'a> TextureBuilder<'a> {
    /// Crate builder with specified kind.
    pub fn new(kind: Kind) -> Self {
        let extent = kind.extent();
        TextureBuilder {
            kind: kind,
            format: Format::Rgba8Srgb,
            data_width: extent.width,
            data_height: extent.height,
            data: Vec::new().into(),
        }
    }

    /// Set image format of the texture to create.
    pub fn with_format(mut self, format: Format) -> Self {
        self.set_format(format);
        self
    }

    /// Set image format of the texture to create.
    pub fn set_format(&mut self, format: Format) -> &mut Self {
        assert_eq!(format.surface_desc().aspects, Aspects::COLOR);
        self.format = format;
        self
    }

    /// Set data width of the raw image bytes (also known as stride).
    /// The number of bytes between lines of the image.
    pub fn with_data_width(mut self, data_width: u32) -> Self {
        self.set_data_width(data_width);
        self
    }

    /// Set data width of the raw image bytes (also known as stride).
    /// The number of bytes between lines of the image.
    pub fn set_data_width(&mut self, data_width: u32) -> &mut Self {
        assert!(data_width >= self.kind.extent().width);
        self.data_width = data_width;
        self
    }

    /// Set data height of the raw image bytes.
    /// The number of bytes between layers of the 3d image.
    pub fn with_data_height(mut self, data_height: u32) -> Self {
        self.set_data_height(data_height);
        self
    }

    /// Set data height of the raw image bytes.
    /// The number of bytes between layers of the 3d image.
    pub fn set_data_height(&mut self, data_height: u32) -> &mut Self {
        assert!(data_height >= self.kind.extent().height);
        self.data_height = data_height;
        self
    }

    /// Set raw data for the image.
    pub fn with_data<D, P>(mut self, data: D) -> Self
    where
        D: Into<Cow<'a, [P]>>,
        P: Clone + 'a,
    {
        self.set_data(data);
        self
    }

    /// Set raw data for the image.
    pub fn set_data<D, P>(&mut self, data: D) -> &mut Self
    where
        D: Into<Cow<'a, [P]>>,
        P: Clone + 'a,
    {
        self.data = cast_cow(data.into());
        self
    }

    /// Build texture and filling it with data provided.
    pub fn build<B>(
        &self,
        family: QueueFamilyId,
        factory: &mut Factory<B>,
    ) -> Result<Texture<B>, Error>
    where
        B: Backend,
    {
        let extent = self.kind.extent();
        assert!(self.data_width >= extent.width);
        assert!(
            self.data.len() * 8
                >= (self.data_width
                    * extent.height
                    * extent.depth
                    * self.format.base_format().0.desc().bits as u32) as usize
        );

        let mut image = factory.create_image(
            self.kind,
            1,
            self.format,
            Tiling::Optimal,
            StorageFlags::empty(),
            Usage::TRANSFER_DST | Usage::SAMPLED,
            Properties::DEVICE_LOCAL,
        )?;

        let view = factory.create_image_view(
            image.borrow(),
            match self.kind {
                Kind::D1(_, _) => ViewKind::D1,
                Kind::D2(_, _, _, _) => ViewKind::D2,
                Kind::D3(_, _, _) => ViewKind::D3,
            },
            self.format,
            Swizzle::NO,
            SubresourceRange {
                aspects: Aspects::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
        )?;

        factory.upload_image(
            &mut image,
            family,
            Layout::ShaderReadOnlyOptimal,
            Access::SHADER_READ,
            SubresourceLayers {
                aspects: Aspects::COLOR,
                level: 0,
                layers: 0..1,
            },
            Offset::ZERO,
            self.kind.extent(),
            self.data_width,
            self.data_height,
            &self.data,
        )?;

        Ok(Texture {
            kind: self.kind,
            format: self.format,
            image,
            view,
        })
    }
}

/// Texture is persistent image accessible by GPU as sampled.
#[derive(Debug)]
pub struct Texture<B: Backend> {
    kind: Kind,
    format: Format,
    image: Image<B>,
    view: B::ImageView,
}

impl<B> Texture<B>
where
    B: Backend,
{
    pub fn new<'a>(kind: Kind) -> TextureBuilder<'a> {
        TextureBuilder::new(kind)
    }

    pub fn image(&self) -> &Image<B> {
        &self.image
    }

    pub fn view(&self) -> &B::ImageView {
        &self.view
    }

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }
}

fn cast_vec<T>(mut vec: Vec<T>) -> Vec<u8> {
    use std::mem;

    let raw_len = mem::size_of::<T>() * vec.len();
    let len = raw_len;

    let cap = mem::size_of::<T>() * vec.capacity();

    let ptr = vec.as_mut_ptr();
    mem::forget(vec);
    unsafe { Vec::from_raw_parts(ptr as _, len, cap) }
}

fn cast_slice<T>(slice: &[T]) -> &[u8] {
    use std::{mem, slice::from_raw_parts};

    let raw_len = mem::size_of::<T>() * slice.len();
    let len = raw_len;

    let ptr = slice.as_ptr();
    mem::forget(slice);
    unsafe { from_raw_parts(ptr as _, len) }
}

fn cast_cow<T>(cow: Cow<[T]>) -> Cow<[u8]>
where
    T: Clone,
{
    match cow {
        Cow::Borrowed(slice) => Cow::Borrowed(cast_slice(slice)),
        Cow::Owned(vec) => Cow::Owned(cast_vec(vec)),
    }
}
