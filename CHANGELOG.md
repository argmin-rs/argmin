# Changelog

## argmin v0.4.6 (18 July 2021)

- updated dependencies (#121, #123, #129, @stefan-k):
  + ndarray 0.15
  + ndarray-linalg 0.14
  + appox 0.5
  + nalgebra 0.28
  + ndarray-rand 0.14
  + num-complex 0.4
  + finitediff 0.1.4

## argmin v0.4.5 

- Squash warnings for Nalgebra 0.26.x (#118, #117, @CattleProdigy)

## argmin v0.4.4 

- Finally started writing a changelog.
- Performance improvements (#111, #112, @sdd)

## argmin v0.4.3

- Downgraded argmin-rand to 0.13 to match ndarray 0.14

## argmin v0.4.2

- Fix lazy evaluation of gradients in line searches (#101, @w1th0utnam3)
- Various updated dependencies

## argmin v0.4.1

- Typo

## argmin 0.4.0

- nalgebra support (#68, @Maher4Ever)
- remove unecessary Default bound on NelderMead (#73, @vadixidav)
- Various updated dependencies

## argmin 0.3.1

- remove finitediff from ndarrayl feature (#61, @optozorax)
- MoreThuente: Added error check for NaN or Inf (#57, @MattBurn)

## argmin 0.3.0

- Golden-section search (#49, @nilgoyette)
- Allow users to choose floating point precision (#50, @stefan-k)
- Remove Clone trait bound from ArgminOp (#48, @stefan-k)
- Remove Serialize trait bound on ArgminOp (#36, @stefan-k)
- Moved from failure to anyhow and thiserror (#44, @stefan-k)
- Added easier access to op and state of ArgminResult (#45, @stefan-k)
- No reexport of argmin_testfunctions (#46, @stefan-k)
- Exposed stopping criterion tolerances of Quasi-Newton methods to user(#43, @stefan-k)
- Exposed stopping criterion tolerance of NewtonCG method to user (@stefan-k)
- Exposed stopping criterion tolerances of Gauss Newton methods to user (@stefan-k)
- Exposed L-BFGS stopping criterion tolerances to user (#37, @stefan-k)
- Removed need for unwrap in cstep MoreThuente LineSearch (#38, @MattBurn)
- Removed Send and Sync trait bounds from ArgminOp (#33, @stefan-k)

## argmin 0.2.6

- Brent's method (#22, @xemwebe)

## argmin 0.2.5

- Particle Swarm Optimization (@jjbayer)
- Derive Clone trait (#14, @rth)
- Test convergence (#13, @rth)
- Lints (#11, @rth)
- Improvements in CG method (@stefan-k)

## argmin 0.2.4

- CG improvments (@stefan-k)

## older versions

For older versions please see the Git history.
