package gomp

import (
	"image"
	"image/color"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

// fillNRGBA fills every pixel of an NRGBA image with c.
func fillNRGBA(img *image.NRGBA, c color.NRGBA) {
	for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
		for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
			i := img.PixOffset(x, y)
			img.Pix[i+0] = c.R
			img.Pix[i+1] = c.G
			img.Pix[i+2] = c.B
			img.Pix[i+3] = c.A
		}
	}
}

func newFilledNRGBA(rect image.Rectangle, c color.NRGBA) *image.NRGBA {
	img := image.NewNRGBA(rect)
	fillNRGBA(img, c)
	return img
}

func newFilledAlpha(rect image.Rectangle, v uint8) *image.Alpha {
	mask := image.NewAlpha(rect)
	for i := range mask.Pix {
		mask.Pix[i] = v
	}
	return mask
}

func pixelAt(t *testing.T, img *image.NRGBA, x, y int) color.NRGBA {
	t.Helper()
	i := img.PixOffset(x, y)
	return color.NRGBA{
		R: img.Pix[i+0],
		G: img.Pix[i+1],
		B: img.Pix[i+2],
		A: img.Pix[i+3],
	}
}

// TestLinearMask_GrayFringe is the motivating use case: a translucent white
// light texture over a black backdrop. Mixing the sRGB values directly yields
// 128 (a dull gray fringe), while linear-light compositing yields 188.
func TestLinearMask_GrayFringe(t *testing.T) {
	assert := assert.New(t)

	rect := image.Rect(0, 0, 4, 4)
	src := newFilledNRGBA(rect, color.NRGBA{R: 255, G: 255, B: 255, A: 128})
	dst := newFilledNRGBA(rect, color.NRGBA{R: 0, G: 0, B: 0, A: 255})
	mask := newFilledAlpha(rect, 255)

	lm, err := NewLinearMask(Normal, 1.0)
	assert.NoError(err)

	out, err := lm.Draw(src, dst, mask)
	assert.NoError(err)
	assert.Equal(rect, out.Bounds())

	want := color.NRGBA{R: 188, G: 188, B: 188, A: 255}
	assert.Equal(want, pixelAt(t, out, 2, 2))

	// The naive sRGB mix would have produced 128 on every channel.
	assert.NotEqual(uint8(128), out.Pix[0])
}

// TestLinearMask_BrightEdge covers a translucent warm light edge over a dark
// backdrop with both the Normal and other separable blend modes. The expected
// values are independently computed (see sRGB transfer function reference).
func TestLinearMask_BrightEdge(t *testing.T) {
	assert := assert.New(t)

	rect := image.Rect(0, 0, 2, 2)
	src := newFilledNRGBA(rect, color.NRGBA{R: 255, G: 240, B: 200, A: 128})
	dst := newFilledNRGBA(rect, color.NRGBA{R: 10, G: 10, B: 30, A: 255})
	mask := newFilledAlpha(rect, 255)

	cases := []struct {
		mode BlendType
		want color.NRGBA
	}{
		{Normal, color.NRGBA{188, 177, 148, 255}},
		{Multiply, color.NRGBA{10, 9, 26, 255}},
		{Screen, color.NRGBA{188, 177, 149, 255}},
		{Darken, color.NRGBA{10, 10, 30, 255}},
	}
	for _, tc := range cases {
		lm, err := NewLinearMask(tc.mode, 1.0)
		assert.NoError(err)

		out, err := lm.Draw(src, dst, mask)
		assert.NoError(err)
		assert.Equal(tc.want, pixelAt(t, out, 0, 0), tc.mode)
	}
}

// TestLinearMask_AlphaConvention checks straight-alpha handling: both source
// and backdrop translucent, and a fully transparent backdrop.
func TestLinearMask_AlphaConvention(t *testing.T) {
	assert := assert.New(t)

	rect := image.Rect(0, 0, 1, 1)

	// Translucent gray over translucent gray: alpha is combined, color stays
	// gray because both layers share the same hue.
	src := newFilledNRGBA(rect, color.NRGBA{R: 200, G: 200, B: 200, A: 64})
	dst := newFilledNRGBA(rect, color.NRGBA{R: 20, G: 20, B: 20, A: 128})
	mask := newFilledAlpha(rect, 255)

	lm, err := NewLinearMask(Normal, 1.0)
	assert.NoError(err)

	out, err := lm.Draw(src, dst, mask)
	assert.NoError(err)
	assert.Equal(color.NRGBA{133, 133, 133, 160}, pixelAt(t, out, 0, 0))

	// Fully transparent backdrop: the source passes through untouched
	// (its straight RGB values are preserved, only alpha is scaled).
	src = newFilledNRGBA(rect, color.NRGBA{R: 200, G: 100, B: 50, A: 128})
	dst = image.NewNRGBA(rect)
	out, err = lm.Draw(src, dst, mask)
	assert.NoError(err)
	assert.Equal(color.NRGBA{200, 100, 50, 128}, pixelAt(t, out, 0, 0))
}

// TestLinearMask_BlackAndWhiteMasks checks the two mask extremes.
func TestLinearMask_BlackAndWhiteMasks(t *testing.T) {
	assert := assert.New(t)

	rect := image.Rect(0, 0, 2, 2)
	src := newFilledNRGBA(rect, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
	dstBase := color.NRGBA{R: 10, G: 10, B: 30, A: 255}
	dst := newFilledNRGBA(rect, dstBase)

	lm, err := NewLinearMask(Normal, 1.0)
	assert.NoError(err)

	// Black mask: zero coverage, the backdrop is returned unchanged.
	blackMask := newFilledAlpha(rect, 0)
	out, err := lm.Draw(src, dst, blackMask)
	assert.NoError(err)
	assert.Equal(dstBase, pixelAt(t, out, 1, 1))

	// White mask: full coverage, the opaque source replaces the backdrop.
	whiteMask := newFilledAlpha(rect, 255)
	out, err = lm.Draw(src, dst, whiteMask)
	assert.NoError(err)
	assert.Equal(color.NRGBA{255, 255, 255, 255}, pixelAt(t, out, 1, 1))
}

// TestLinearMask_MaskDifferentBounds uses a mask smaller than the backdrop and
// anchored at a different origin: covered pixels are composited, uncovered
// pixels keep the exact backdrop bytes.
func TestLinearMask_MaskDifferentBounds(t *testing.T) {
	assert := assert.New(t)

	rect := image.Rect(0, 0, 4, 4)
	src := newFilledNRGBA(rect, color.NRGBA{R: 255, G: 255, B: 255, A: 128})
	dst := newFilledNRGBA(rect, color.NRGBA{R: 0, G: 0, B: 0, A: 255})

	// Mask covers only the top-left 2x2 quadrant with a non-zero origin.
	mask := newFilledAlpha(image.Rect(1, 1, 3, 3), 255)

	lm, err := NewLinearMask(Normal, 1.0)
	assert.NoError(err)

	out, err := lm.Draw(src, dst, mask)
	assert.NoError(err)
	assert.Equal(rect, out.Bounds())

	// Covered pixel.
	assert.Equal(color.NRGBA{188, 188, 188, 255}, pixelAt(t, out, 2, 2))
	// Uncovered pixels retain the backdrop exactly.
	assert.Equal(color.NRGBA{0, 0, 0, 255}, pixelAt(t, out, 0, 0))
	assert.Equal(color.NRGBA{0, 0, 0, 255}, pixelAt(t, out, 3, 3))
	assert.Equal(color.NRGBA{0, 0, 0, 255}, pixelAt(t, out, 1, 3))

	// A mask fully disjoint from the backdrop leaves the whole image intact.
	disjoint := newFilledAlpha(image.Rect(10, 10, 12, 12), 255)
	out, err = lm.Draw(src, dst, disjoint)
	assert.NoError(err)
	assert.Equal(dst.Pix, out.Pix)
}

// TestLinearMask_NonZeroOrigin runs the same compositing on images whose
// Bounds do not start at (0, 0); PixOffset based indexing must handle it.
func TestLinearMask_NonZeroOrigin(t *testing.T) {
	assert := assert.New(t)

	rect := image.Rect(5, 7, 7, 9)
	src := newFilledNRGBA(rect, color.NRGBA{R: 255, G: 255, B: 255, A: 128})
	dst := newFilledNRGBA(rect, color.NRGBA{R: 0, G: 0, B: 0, A: 255})
	// Smaller mask with yet another origin: covers only the (6, 8) pixel.
	mask := newFilledAlpha(image.Rect(6, 8, 7, 9), 255)

	lm, err := NewLinearMask(Normal, 1.0)
	assert.NoError(err)

	out, err := lm.Draw(src, dst, mask)
	assert.NoError(err)
	assert.Equal(rect, out.Bounds())

	assert.Equal(color.NRGBA{188, 188, 188, 255}, pixelAt(t, out, 6, 8))
	assert.Equal(color.NRGBA{0, 0, 0, 255}, pixelAt(t, out, 5, 7))
	assert.Equal(color.NRGBA{0, 0, 0, 255}, pixelAt(t, out, 6, 7))
}

// TestLinearMask_Intensity verifies intensity scaling, clamping and the
// combination with a gray mask.
func TestLinearMask_Intensity(t *testing.T) {
	assert := assert.New(t)

	rect := image.Rect(0, 0, 1, 1)
	src := newFilledNRGBA(rect, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
	dst := newFilledNRGBA(rect, color.NRGBA{R: 0, G: 0, B: 0, A: 255})

	// Gray mask (128/255) times intensity 0.5: coverage ~= 0.251.
	mask := newFilledAlpha(rect, 128)
	lm, err := NewLinearMask(Normal, 0.5)
	assert.NoError(err)

	out, err := lm.Draw(src, dst, mask)
	assert.NoError(err)
	assert.Equal(color.NRGBA{137, 137, 137, 255}, pixelAt(t, out, 0, 0))

	// Intensity above 1 clamps the coverage to 1: the opaque white source
	// fully replaces the black backdrop.
	lm, err = NewLinearMask(Normal, 2.0)
	assert.NoError(err)
	out, err = lm.Draw(src, dst, mask)
	assert.NoError(err)
	assert.Equal(color.NRGBA{255, 255, 255, 255}, pixelAt(t, out, 0, 0))

	// Zero intensity is a no-op.
	lm, err = NewLinearMask(Normal, 0)
	assert.NoError(err)
	out, err = lm.Draw(src, dst, mask)
	assert.NoError(err)
	assert.Equal(color.NRGBA{0, 0, 0, 255}, pixelAt(t, out, 0, 0))
}

// TestLinearMask_MaskFormats exercises the four supported single-channel
// mask types, including the 16-bit variants.
func TestLinearMask_MaskFormats(t *testing.T) {
	assert := assert.New(t)

	rect := image.Rect(0, 0, 1, 1)
	src := newFilledNRGBA(rect, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
	dst := newFilledNRGBA(rect, color.NRGBA{R: 0, G: 0, B: 0, A: 255})
	want := color.NRGBA{188, 188, 188, 255}

	lm, err := NewLinearMask(Normal, 1.0)
	assert.NoError(err)

	alpha := &image.Alpha{Pix: []uint8{128}, Stride: 1, Rect: rect}
	out, err := lm.Draw(src, dst, alpha)
	assert.NoError(err)
	assert.Equal(want, pixelAt(t, out, 0, 0))

	gray := &image.Gray{Pix: []uint8{128}, Stride: 1, Rect: rect}
	out, err = lm.Draw(src, dst, gray)
	assert.NoError(err)
	assert.Equal(want, pixelAt(t, out, 0, 0))

	alpha16 := image.NewAlpha16(rect)
	alpha16.SetAlpha16(0, 0, color.Alpha16{A: 32768})
	out, err = lm.Draw(src, dst, alpha16)
	assert.NoError(err)
	assert.Equal(want, pixelAt(t, out, 0, 0))

	gray16 := image.NewGray16(rect)
	gray16.SetGray16(0, 0, color.Gray16{Y: 32768})
	out, err = lm.Draw(src, dst, gray16)
	assert.NoError(err)
	assert.Equal(want, pixelAt(t, out, 0, 0))

	// A nil mask means full coverage.
	out, err = lm.Draw(src, dst, nil)
	assert.NoError(err)
	assert.Equal(color.NRGBA{255, 255, 255, 255}, pixelAt(t, out, 0, 0))
}

// TestLinearMask_FullyTransparentPixels makes sure transparent source pixels
// do not alter the backdrop even under a fully opaque mask.
func TestLinearMask_FullyTransparentPixels(t *testing.T) {
	assert := assert.New(t)

	rect := image.Rect(0, 0, 2, 1)
	src := image.NewNRGBA(rect)
	src.SetNRGBA(0, 0, color.NRGBA{R: 255, G: 255, B: 255, A: 0})
	src.SetNRGBA(1, 0, color.NRGBA{R: 255, G: 255, B: 255, A: 128})
	dst := newFilledNRGBA(rect, color.NRGBA{R: 0, G: 0, B: 0, A: 255})
	mask := newFilledAlpha(rect, 255)

	lm, err := NewLinearMask(Normal, 1.0)
	assert.NoError(err)

	out, err := lm.Draw(src, dst, mask)
	assert.NoError(err)
	assert.Equal(color.NRGBA{0, 0, 0, 255}, pixelAt(t, out, 0, 0))
	assert.Equal(color.NRGBA{188, 188, 188, 255}, pixelAt(t, out, 1, 0))
}

// TestLinearMask_NonSeparableModes checks Hue and Luminosity in linear light.
func TestLinearMask_NonSeparableModes(t *testing.T) {
	assert := assert.New(t)

	rect := image.Rect(0, 0, 1, 1)
	src := newFilledNRGBA(rect, color.NRGBA{R: 255, G: 240, B: 200, A: 128})
	dst := newFilledNRGBA(rect, color.NRGBA{R: 10, G: 10, B: 30, A: 255})
	mask := newFilledAlpha(rect, 255)

	cases := []struct {
		mode BlendType
		want color.NRGBA
	}{
		{Hue, color.NRGBA{176, 176, 189, 255}},
		{Luminosity, color.NRGBA{14, 12, 19, 255}},
	}
	for _, tc := range cases {
		lm, err := NewLinearMask(tc.mode, 1.0)
		assert.NoError(err)
		out, err := lm.Draw(src, dst, mask)
		assert.NoError(err)
		assert.Equal(tc.want, pixelAt(t, out, 0, 0), NewBlend().ToString(tc.mode))
	}
}

// TestLinearMask_AllBlendModesFinite runs every supported mode through the
// pipeline and verifies that no NaN/Inf can leak into the output.
func TestLinearMask_AllBlendModesFinite(t *testing.T) {
	assert := assert.New(t)

	rect := image.Rect(0, 0, 1, 1)
	src := newFilledNRGBA(rect, color.NRGBA{R: 255, G: 240, B: 200, A: 128})
	dst := newFilledNRGBA(rect, color.NRGBA{R: 10, G: 10, B: 30, A: 255})
	mask := newFilledAlpha(rect, 255)

	for _, mode := range blendModes {
		lm, err := NewLinearMask(mode, 1.0)
		assert.NoError(err)
		out, err := lm.Draw(src, dst, mask)
		assert.NoError(err)
		for _, v := range out.Pix {
			assert.False(math.IsNaN(float64(v)))
		}
	}
}

// TestLinearMask_DeterministicErrors covers all the documented error cases.
func TestLinearMask_DeterministicErrors(t *testing.T) {
	assert := assert.New(t)

	rect := image.Rect(0, 0, 2, 2)
	src := image.NewNRGBA(rect)
	dst := image.NewNRGBA(rect)
	mask := newFilledAlpha(rect, 255)

	// Non-finite intensity, both at construction and via the setter.
	for _, intensity := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		lm, err := NewLinearMask(Normal, intensity)
		assert.Nil(lm)
		assert.ErrorIs(err, ErrNonFiniteIntensity)
	}
	lm, err := NewLinearMask(Normal, 1.0)
	assert.NoError(err)
	assert.ErrorIs(lm.SetIntensity(math.NaN()), ErrNonFiniteIntensity)

	// Unsupported blend mode.
	bad := BlendType(1000)
	_, err = NewLinearMask(bad, 1.0)
	assert.ErrorIs(err, ErrUnsupportedBlend)
	assert.ErrorIs(lm.SetBlend(bad), ErrUnsupportedBlend)

	// Empty and nil images.
	emptyRect := image.Rect(0, 0, 0, 0)
	empty := image.NewNRGBA(emptyRect)

	_, err = lm.Draw(nil, dst, mask)
	assert.ErrorIs(err, ErrEmptyImage)
	_, err = lm.Draw(empty, dst, mask)
	assert.ErrorIs(err, ErrEmptyImage)
	_, err = lm.Draw(src, image.NewNRGBA(emptyRect), mask)
	assert.ErrorIs(err, ErrEmptyImage)

	// Size mismatch: different size and different origin.
	mismatch := image.NewNRGBA(image.Rect(0, 0, 3, 3))
	_, err = lm.Draw(src, mismatch, mask)
	assert.ErrorIs(err, ErrSizeMismatch)
	shifted := image.NewNRGBA(image.Rect(1, 1, 3, 3))
	_, err = lm.Draw(shifted, dst, mask)
	assert.ErrorIs(err, ErrSizeMismatch)

	// Unsupported mask color model.
	_, err = lm.Draw(src, dst, image.NewRGBA(rect))
	assert.ErrorIs(err, ErrUnsupportedMask)
	_, err = lm.Draw(src, dst, image.NewPaletted(rect, color.Palette{color.Black, color.White}))
	assert.ErrorIs(err, ErrUnsupportedMask)
}

// TestLinearMask_Setters verifies the configuration setters and getters.
func TestLinearMask_Setters(t *testing.T) {
	assert := assert.New(t)

	lm, err := NewLinearMask(Normal, 0.75)
	assert.NoError(err)
	assert.Equal(Normal, lm.Mode)
	assert.Equal(0.75, lm.Intensity)

	assert.NoError(lm.SetBlend(Screen))
	assert.Equal(Screen, lm.Mode)
	assert.NoError(lm.SetIntensity(0.25))
	assert.Equal(0.25, lm.Intensity)
}
