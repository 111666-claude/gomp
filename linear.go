package gomp

import (
	"errors"
	"image"
	"math"
)

// Sentinel errors returned by the linear-light mask composition.
// They are returned wrapped inside the error value of LinearMask.Draw.
var (
	// ErrEmptyImage is returned when the source or the backdrop is nil
	// or has an empty Bounds rectangle.
	ErrEmptyImage = errors.New("gomp: source and backdrop images must be non-nil with non-empty bounds")

	// ErrSizeMismatch is returned when the source and the backdrop do not
	// share the exact same Bounds.
	ErrSizeMismatch = errors.New("gomp: source and backdrop bounds must match")

	// ErrNonFiniteIntensity is returned when the intensity is not a finite
	// floating point value (NaN, +Inf or -Inf).
	ErrNonFiniteIntensity = errors.New("gomp: intensity must be a finite value")

	// ErrUnsupportedMask is returned when the mask color model cannot provide
	// a single channel (alpha or gray) value.
	ErrUnsupportedMask = errors.New("gomp: unsupported mask color model, use *image.Alpha, *image.Alpha16, *image.Gray or *image.Gray16")

	// ErrUnsupportedBlend is returned when the requested blend mode is not supported.
	ErrUnsupportedBlend = errors.New("gomp: unsupported blend mode")
)

// LinearMask composites a source image over a backdrop using a single-channel
// mask and an intensity factor, performing all color math in linear-light
// (linearized sRGB) space before writing the result back as sRGB.
//
// Blending translucent light textures directly on sRGB encoded values darkens
// the soft edges and produces visible gray fringes. Linearizing the colors
// first makes the compositing physically correct.
//
// The output always has the same Bounds as the backdrop. A nil mask applies
// full coverage everywhere; otherwise pixels outside the mask Bounds keep the
// backdrop unchanged, and a mask value of zero leaves them untouched as well.
type LinearMask struct {
	// Mode selects the blend mode applied between the source and the backdrop.
	// Normal reproduces a source-over light source.
	Mode BlendType

	// Intensity scales the effective mask coverage. Values outside the [0, 1]
	// range are clamped; a non-finite value makes Draw return an error.
	Intensity float64
}

// NewLinearMask instantiates a LinearMask using the provided blend mode and
// intensity. It returns an error when the blend mode is not supported or the
// intensity is not finite.
func NewLinearMask(mode BlendType, intensity float64) (*LinearMask, error) {
	lm := &LinearMask{Mode: mode, Intensity: intensity}
	if err := lm.validate(); err != nil {
		return nil, err
	}
	return lm, nil
}

// SetBlend changes the active blend mode.
func (lm *LinearMask) SetBlend(mode BlendType) error {
	if !isValidBlendMode(mode) {
		return ErrUnsupportedBlend
	}
	lm.Mode = mode
	return nil
}

// SetIntensity changes the mask intensity. Non-finite values are rejected.
func (lm *LinearMask) SetIntensity(intensity float64) error {
	if !isFinite(intensity) {
		return ErrNonFiniteIntensity
	}
	lm.Intensity = intensity
	return nil
}

// validate checks the blend mode and the intensity.
func (lm *LinearMask) validate() error {
	if !isValidBlendMode(lm.Mode) {
		return ErrUnsupportedBlend
	}
	if !isFinite(lm.Intensity) {
		return ErrNonFiniteIntensity
	}
	return nil
}

// Draw composites src over dst through mask using the configured blend mode
// and intensity. The color channels are converted from sRGB to linear light
// before compositing and encoded back to sRGB afterwards.
//
// The returned *image.NRGBA has exactly dst.Bounds(). Areas not covered by the
// mask keep the backdrop pixels unchanged. An error is returned for nil/empty
// images, mismatched source and backdrop Bounds, non-finite intensity or an
// unsupported mask color model; in those cases no partial output is produced.
func (lm *LinearMask) Draw(src, dst *image.NRGBA, mask image.Image) (*image.NRGBA, error) {
	if err := lm.validate(); err != nil {
		return nil, err
	}
	if src == nil || dst == nil || src.Bounds().Empty() || dst.Bounds().Empty() {
		return nil, ErrEmptyImage
	}
	if src.Bounds() != dst.Bounds() {
		return nil, ErrSizeMismatch
	}
	if mask != nil && !isSupportedMask(mask) {
		return nil, ErrUnsupportedMask
	}

	out := image.NewNRGBA(dst.Bounds())
	copy(out.Pix, dst.Pix)

	bounds := dst.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			coverage, covered := maskCoverage(mask, x, y)
			if !covered {
				continue // outside the mask Bounds: keep the backdrop untouched
			}
			coverage = clamp01(coverage * lm.Intensity)
			if coverage == 0 {
				continue
			}

			si := src.PixOffset(x, y)
			di := out.PixOffset(x, y)

			// Premultiplied source alpha after the mask: a fully transparent
			// source pixel cannot modify the backdrop.
			as := float64(src.Pix[si+3]) / 255 * coverage
			if as == 0 {
				continue
			}
			ab := float64(dst.Pix[di+3]) / 255

			ao := as + ab*(1-as)
			// ao is strictly positive here: as > 0 implies ao > 0.
			cs := Color{
				R: srgbToLinear(float64(src.Pix[si+0]) / 255),
				G: srgbToLinear(float64(src.Pix[si+1]) / 255),
				B: srgbToLinear(float64(src.Pix[si+2]) / 255),
			}
			cb := Color{
				R: srgbToLinear(float64(dst.Pix[di+0]) / 255),
				G: srgbToLinear(float64(dst.Pix[di+1]) / 255),
				B: srgbToLinear(float64(dst.Pix[di+2]) / 255),
			}

			var blended Color
			if lm.isNonSeparable() {
				blended = lm.blendColor(cs, cb)
			} else {
				blended = Color{
					R: lm.blendChannel(cs.R, cb.R),
					G: lm.blendChannel(cs.G, cb.G),
					B: lm.blendChannel(cs.B, cb.B),
				}
			}

			// W3C compositing formula with non-premultiplied (straight)
			// RGB and the source-over alpha convention.
			nr := (as*(1-ab)*cs.R + ab*as*blended.R + (1-as)*ab*cb.R) / ao
			ng := (as*(1-ab)*cs.G + ab*as*blended.G + (1-as)*ab*cb.G) / ao
			nb := (as*(1-ab)*cs.B + ab*as*blended.B + (1-as)*ab*cb.B) / ao

			out.Pix[di+0] = linearToSRGB8(nr)
			out.Pix[di+1] = linearToSRGB8(ng)
			out.Pix[di+2] = linearToSRGB8(nb)
			out.Pix[di+3] = uint8(math.Round(ao * 255))
		}
	}

	return out, nil
}

// isNonSeparable reports whether the active mode operates on the whole color
// at once instead of on independent channels.
func (lm *LinearMask) isNonSeparable() bool {
	switch lm.Mode {
	case Hue, Saturation, ColorMode, Luminosity:
		return true
	}
	return false
}

// blendChannel applies the separable blend function of the active mode to one
// color channel. Non-separable modes are handled by blendColor instead.
func (lm *LinearMask) blendChannel(cs, cb float64) float64 {
	switch lm.Mode {
	case Normal:
		return cs
	case Darken:
		return min(cs, cb)
	case Lighten:
		return max(cs, cb)
	case Multiply:
		return cs * cb
	case Screen:
		return 1 - (1-cs)*(1-cb)
	case Overlay:
		if cb <= 0.5 {
			return 2 * cs * cb
		}
		return 1 - 2*(1-cs)*(1-cb)
	case SoftLight:
		if cb < 0.5 {
			return cs - (1-2*cb)*cs*(1-cs)
		}
		var d float64
		if cs < 0.25 {
			d = ((16*cs-12)*cs + 4) * cs
		} else {
			d = math.Sqrt(cs)
		}
		return cs + (2*cb-1)*(d-cs)
	case HardLight:
		if cs <= 0.5 {
			return 2 * cs * cb
		}
		return 1 - 2*(1-cs)*(1-cb)
	case ColorDodge:
		if cs >= 1 {
			return 1
		}
		return min(1, cb/(1-cs))
	case ColorBurn:
		if cs <= 0 {
			return 0
		}
		return 1 - min(1, (1-cb)/cs)
	case Difference:
		return abs(cb - cs)
	case Exclusion:
		return cs + cb - 2*cs*cb
	}
	// Non-separable modes never reach this function.
	return cs
}

// blendColor applies the non-separable blend modes (Hue, Saturation, ColorMode,
// Luminosity) in linear light. The existing Blend helpers are color-space
// agnostic and operate on normalized [0, 1] values.
func (lm *LinearMask) blendColor(cs, cb Color) Color {
	bl := NewBlend()
	switch lm.Mode {
	case Hue:
		sat := bl.SetSat(cb, bl.Sat(cs))
		return bl.SetLum(sat, bl.Lum(cs))
	case Saturation:
		sat := bl.SetSat(cs, bl.Sat(cb))
		return bl.SetLum(sat, bl.Lum(cs))
	case ColorMode:
		return bl.SetLum(cb, bl.Lum(cs))
	case Luminosity:
		return bl.SetLum(cs, bl.Lum(cb))
	}
	return cs
}

// isSupportedMask reports whether mask provides a single channel value.
func isSupportedMask(mask image.Image) bool {
	switch mask.(type) {
	case *image.Alpha, *image.Alpha16, *image.Gray, *image.Gray16:
		return true
	}
	return false
}

// maskCoverage returns the normalized [0, 1] coverage of mask at (x, y).
// The second return value is false when the point lies outside the mask
// Bounds, in which case the caller must keep the backdrop untouched. A nil
// mask means full coverage everywhere.
func maskCoverage(mask image.Image, x, y int) (float64, bool) {
	if mask == nil {
		return 1, true
	}
	if !(image.Point{x, y}.In(mask.Bounds())) {
		return 0, false
	}
	switch m := mask.(type) {
	case *image.Alpha:
		return float64(m.AlphaAt(x, y).A) / 255, true
	case *image.Alpha16:
		return float64(m.Alpha16At(x, y).A) / 65535, true
	case *image.Gray:
		return float64(m.GrayAt(x, y).Y) / 255, true
	case *image.Gray16:
		return float64(m.Gray16At(x, y).Y) / 65535, true
	}
	return 0, false
}

// srgbToLinear converts a normalized sRGB encoded value to linear light using
// the IEC 61966-2-1 transfer function.
func srgbToLinear(c float64) float64 {
	if c <= 0.04045 {
		return c / 12.92
	}
	return math.Pow((c+0.055)/1.055, 2.4)
}

// linearToSRGB converts a normalized linear-light value back to sRGB.
func linearToSRGB(c float64) float64 {
	c = clamp01(c)
	if c <= 0.0031308 {
		return 12.92 * c
	}
	return 1.055*math.Pow(c, 1/2.4) - 0.055
}

// linearToSRGB8 encodes a linear value to sRGB and rounds it to an uint8.
func linearToSRGB8(c float64) uint8 {
	return uint8(math.Round(clamp01(linearToSRGB(c)) * 255))
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}
