/**
 * --------------------------------------------------------------------------
 * Bootstrap (v5.0.0-beta3): carousel.js
 * Licensed under MIT (https://github.com/twbs/bootstrap/blob/main/LICENSE)
 * --------------------------------------------------------------------------
 */

import {
  defineJQueryPlugin,
  emulateTransitionEnd,
  getElementFromSelector,
  getTransitionDurationFromElement,
  isRTL,
  isVisible,
  reflow,
  triggerTransitionEnd,
  typeCheckConfig
} from './util/index'
import Data from './dom/data'
import EventHandler from './dom/event-handler'
import Manipulator from './dom/manipulator'
import SelectorEngine from './dom/selector-engine'
import BaseComponent from './base-component'

/**
 * ------------------------------------------------------------------------
 * Constants
 * ------------------------------------------------------------------------
 */

const NAME = 'carousel'
const DATA_KEY = 'bs.carousel'
const EVENT_KEY = `.${DATA_KEY}`
const DATA_API_KEY = '.data-api'

const ARROW_LEFT_KEY = 'ArrowLeft'
const ARROW_RIGHT_KEY = 'ArrowRight'
const TOUCHEVENT_COMPAT_WAIT = 500 // Time for mouse compat events to fire after touch
const SWIPE_THRESHOLD = 40

const Default = {
  interval: 5000,
  keyboard: true,
  slide: false,
  pause: 'hover',
  wrap: true,
  touch: true
}

const DefaultType = {
  interval: '(number|boolean)',
  keyboard: 'boolean',
  slide: '(boolean|string)',
  pause: '(string|boolean)',
  wrap: 'boolean',
  touch: 'boolean'
}

const ORDER_NEXT = 'next'
const ORDER_PREV = 'prev'
const DIRECTION_LEFT = 'left'
const DIRECTION_RIGHT = 'right'

const EVENT_SLIDE = `slide${EVENT_KEY}`
const EVENT_SLID = `slid${EVENT_KEY}`
const EVENT_KEYDOWN = `keydown${EVENT_KEY}`
const EVENT_MOUSEENTER = `mouseenter${EVENT_KEY}`
const EVENT_MOUSELEAVE = `mouseleave${EVENT_KEY}`
const EVENT_TOUCHSTART = `touchstart${EVENT_KEY}`
const EVENT_TOUCHMOVE = `touchmove${EVENT_KEY}`
const EVENT_TOUCHEND = `touchend${EVENT_KEY}`
const EVENT_POINTERDOWN = `pointerdown${EVENT_KEY}`
const EVENT_POINTERUP = `pointerup${EVENT_KEY}`
const EVENT_DRAG_START = `dragstart${EVENT_KEY}`
const EVENT_LOAD_DATA_API = `load${EVENT_KEY}${DATA_API_KEY}`
const EVENT_CLICK_DATA_API = `click${EVENT_KEY}${DATA_API_KEY}`

const CLASS_NAME_CAROUSEL = 'carousel'
const CLASS_NAME_ACTIVE = 'active'
const CLASS_NAME_SLIDE = 'slide'
const CLASS_NAME_END = 'carousel-item-end'
const CLASS_NAME_START = 'carousel-item-start'
const CLASS_NAME_NEXT = 'carousel-item-next'
const CLASS_NAME_PREV = 'carousel-item-prev'
const CLASS_NAME_POINTER_EVENT = 'pointer-event'

const SELECTOR_ACTIVE = '.active'
const SELECTOR_ACTIVE_ITEM = '.active.carousel-item'
const SELECTOR_ITEM = '.carousel-item'
const SELECTOR_ITEM_IMG = '.carousel-item img'
const SELECTOR_NEXT_PREV = '.carousel-item-next, .carousel-item-prev'
const SELECTOR_INDICATORS = '.carousel-indicators'
const SELECTOR_INDICATOR = '[data-bs-target]'
const SELECTOR_DATA_SLIDE = '[data-bs-slide], [data-bs-slide-to]'
const SELECTOR_DATA_RIDE = '[data-bs-ride="carousel"]'

const POINTER_TYPE_TOUCH = 'touch'
const POINTER_TYPE_PEN = 'pen'

/**
 * ------------------------------------------------------------------------
 * Class Definition
 * ------------------------------------------------------------------------
 */
class Carousel extends BaseComponent {
  constructor(element, config) {
    super(element)

    this._items = null
    this._interval = null
    this._activeElement = null
    this._isPaused = false
    this._isSliding = false
    this.touchTimeout = null
    this.touchStartX = 0
    this.touchDeltaX = 0

    this._config = this._getConfig(config)
    this._indicatorsElement = SelectorEngine.findOne(SELECTOR_INDICATORS, this._element)
    this._touchSupported = 'ontouchstart' in document.documentElement || navigator.maxTouchPoints > 0
    this._pointerEvent = Boolean(window.PointerEvent)

    this._addEventListeners()
  }

  // Getters

  static get Default() {
    return Default
  }

  static get DATA_KEY() {
    return DATA_KEY
  }

  // Public

  next() {
    if (!this._isSliding) {
      this._slide(ORDER_NEXT)
    }
  }

  nextWhenVisible() {
    // Don't call next when the page isn't visible
    // or the carousel or its parent isn't visible
    if (!document.hidden && isVisible(this._element)) {
      this.next()
    }
  }

  prev() {
    if (!this._isSliding) {
      this._slide(ORDER_PREV)
    }
  }

  pause(event) {
    if (!event) {
      this._isPaused = true
    }

    if (SelectorEngine.findOne(SELECTOR_NEXT_PREV, this._element)) {
      triggerTransitionEnd(this._element)
      this.cycle(true)
    }

    clearInterval(this._interval)
    this._interval = null
  }

  cycle(event) {
    if (!event) {
      this._isPaused = false
    }

    if (this._interval) {
      clearInterval(this._interval)
      this._interval = null
    }

    if (this._config && this._config.interval && !this._isPaused) {
      this._updateInterval()

      this._interval = setInterval(
        (document.visibilityState ? this.nextWhenVisible : this.next).bind(this),
        this._config.interval
      )
    }
  }

  to(index) {
    this._activeElement = SelectorEngine.findOne(SELECTOR_ACTIVE_ITEM, this._element)
    const activeIndex = this._getItemIndex(this._activeElement)

    if (index > this._items.length - 1 || index < 0) {
      return
    }

    if (this._isSliding) {
      EventHandler.one(this._element, EVENT_SLID, () => this.to(index))
      return
    }

    if (activeIndex === index) {
      this.pause()
      this.cycle()
      return
    }

    const order = index > activeIndex ?
      ORDER_NEXT :
      ORDER_PREV

    this._slide(order, this._items[index])
  }

  dispose() {
    EventHandler.off(this._element, EVENT_KEY)

    this._items = null
    this._config = null
    this._interval = null
    this._isPaused = null
    this._isSliding = null
    this._activeElement = null
    this._indicatorsElement = null

    super.dispose()
  }

  // Private

  _getConfig(config) {
    config = {
      ...Default,
      ...config
    }
    typeCheckConfig(NAME, config, DefaultType)
    return config
  }

  _handleSwipe() {
    const absDeltax = Math.abs(this.touchDeltaX)

    if (absDeltax <= SWIPE_THRESHOLD) {
      return
    }

    const direction = absDeltax / this.touchDeltaX

    this.touchDeltaX = 0

    if (!direction) {
      return
    }

    this._slide(direction > 0 ? DIRECTION_RIGHT : DIRECTION_LEFT)
  }

  _addEventListeners() {
    if (this._config.keyboard) {
      EventHandler.on(this._element, EVENT_KEYDOWN, event => this._keydown(event))
    }

    if (this._config.pause === 'hover') {
      EventHandler.on(this._element, EVENT_MOUSEENTER, event => this.pause(event))
      EventHandler.on(this._element, EVENT_MOUSELEAVE, event => this.cycle(event))
    }

    if (this._config.touch && this._touchSupported) {
      this._addTouchEventListeners()
    }
  }

  _addTouchEventListeners() {
    const start = event => {
      if (this._pointerEvent && (event.pointerType === POINTER_TYPE_PEN || event.pointerType === POINTER_TYPE_TOUCH)) {
        this.touchStartX = event.clientX
      } else if (!this._pointerEvent) {
        this.touchStartX = event.touches[0].clientX
      }
    }

    const move = event => {
      // ensure swiping with one touch and not pinching
      this.touchDeltaX = event.touches && event.touches.length > 1 ?
        0 :
        event.touches[0].clientX - this.touchStartX
    }

    const end = event => {
      if (this._pointerEvent && (event.pointerType === POINTER_TYPE_PEN || event.pointerType === POINTER_TYPE_TOUCH)) {
        this.touchDeltaX = event.clientX - this.touchStartX
      }

      this._handleSwipe()
      if (this._config.pause === 'hover') {
        // If it's a touch-enabled device, mouseenter/leave are fired as
        // part of the mouse compatibility events on first tap - the carousel
        // would stop cycling until user tapped out of it;
        // here, we listen for touchend, explicitly pause the carousel
        // (as if it's the second time we tap on it, mouseenter compat event
        // is NOT fired) and after a timeout (to allow for mouse compatibility
        // events to fire) we explicitly restart cycling

        this.pause()
        if (this.touchTimeout) {
          clearTimeout(this.touchTimeout)
        }

        this.touchTimeout = setTimeout(event => this.cycle(event), TOUCHEVENT_COMPAT_WAIT + this._config.interval)
      }
    }

    SelectorEngine.find(SELECTOR_ITEM_IMG, this._element).forEach(itemImg => {
      EventHandler.on(itemImg, EVENT_DRAG_START, e => e.preventDefault())
    })

    if (this._pointerEvent) {
      EventHandler.on(this._element, EVENT_POINTERDOWN, event => start(event))
      EventHandler.on(this._element, EVENT_POINTERUP, event => end(event))

      this._element.classList.add(CLASS_NAME_POINTER_EVENT)
    } else {
      EventHandler.on(this._element, EVENT_TOUCHSTART, event => start(event))
      EventHandler.on(this._element, EVENT_TOUCHMOVE, event => move(event))
      EventHandler.on(this._element, EVENT_TOUCHEND, event => end(event))
    }
  }

  _keydown(event) {
    if (/input|textarea/i.test(event.target.tagName)) {
      return
    }

    if (event.key === ARROW_LEFT_KEY) {
      event.preventDefault()
      this._slide(DIRECTION_LEFT)
    } else if (event.key === ARROW_RIGHT_KEY) {
      event.preventDefault()
      this._slide(DIRECTION_RIGHT)
    }
  }

  _getItemIndex(element) {
    this._items = element && element.parentNode ?
      SelectorEngine.find(SELECTOR_ITEM, element.parentNode) :
      []

    return this._items.indexOf(element)
  }

  _getItemByOrder(order, activeElement) {
    const isNext = order === ORDER_NEXT
    const isPrev = order === ORDER_PREV
    const activeIndex = this._getItemIndex(activeElement)
    const lastItemIndex = this._items.length - 1
    const isGoingToWrap = (isPrev && activeIndex === 0) || (isNext && activeIndex === lastItemIndex)

    if (isGoingToWrap && !this._config.wrap) {
      return activeElement
    }

    const delta = isPrev ? -1 : 1
    const itemIndex = (activeIndex + delta) % this._items.length

    return itemIndex === -1 ?
      this._items[this._items.length - 1] :
      this._items[itemIndex]
  }

  _triggerSlideEvent(relatedTarget, eventDirectionName) {
    const targetIndex = this._getItemIndex(relatedTarget)
    const fromIndex = this._getItemIndex(SelectorEngine.findOne(SELECTOR_ACTIVE_ITEM, this._element))

    return EventHandler.trigger(this._element, EVENT_SLIDE, {
      relatedTarget,
      direction: eventDirectionName,
      from: fromIndex,
      to: targetIndex
    })
  }

  _setActiveIndicatorElement(element) {
    if (this._indicatorsElement) {
      const activeIndicator = SelectorEngine.findOne(SELECTOR_ACTIVE, this._indicatorsElement)

      activeIndicator.classList.remove(CLASS_NAME_ACTIVE)
      activeIndicator.removeAttribute('aria-current')

      const indicators = SelectorEngine.find(SELECTOR_INDICATOR, this._indicatorsElement)

      for (let i = 0; i < indicators.length; i++) {
        if (Number.parseInt(indicators[i].getAttribute('data-bs-slide-to'), 10) === this._getItemIndex(element)) {
          indicators[i].classList.add(CLASS_NAME_ACTIVE)
          indicators[i].setAttribute('aria-current', 'true')
          break
        }
      }
    }
  }

  _updateInterval() {
    const element = this._activeElement || SelectorEngine.findOne(SELECTOR_ACTIVE_ITEM, this._element)

    if (!element) {
      return
    }

    const elementInterval = Number.parseInt(element.getAttribute('data-bs-interval'), 10)

    if (elementInterval) {
      this._config.defaultInterval = this._config.defaultInterval || this._config.interval
      this._config.interval = elementInterval
    } else {
      this._config.interval = this._config.defaultInterval || this._config.interval
    }
  }

  _slide(directionOrOrder, element) {
    const order = this._directionToOrder(directionOrOrder)
    const activeElement = SelectorEngine.findOne(SELECTOR_ACTIVE_ITEM, this._element)
    const activeElementIndex = this._getItemIndex(activeElement)
    const nextElement = element || this._getItemByOrder(order, activeElement)

    const nextElementIndex = this._getItemIndex(nextElement)
    const isCycling = Boolean(this._interval)

    const isNext = order === ORDER_NEXT
    const directionalClassName = isNext ? CLASS_NAME_START : CLASS_NAME_END
    const orderClassName = isNext ? CLASS_NAME_NEXT : CLASS_NAME_PREV
    const eventDirectionName = this._orderToDirection(order)

    if (nextElement && nextElement.classList.contains(CLASS_NAME_ACTIVE)) {
      this._isSliding = false
      return
    }

    const slideEvent = this._triggerSlideEvent(nextElement, eventDirectionName)
    if (slideEvent.defaultPrevented) {
      return
    }

    if (!activeElement || !nextElement) {
      // Some weirdness is happening, so we bail
      return
    }

    this._isSliding = true

    if (isCycling) {
      this.pause()
    }

    this._setActiveIndicatorElement(nextElement)
    this._activeElement = nextElement

    if (this._element.classList.contains(CLASS_NAME_SLIDE)) {
      nextElement.classList.add(orderClassName)

      reflow(nextElement)

      activeElement.classList.add(directionalClassName)
      nextElement.classList.add(directionalClassName)

      const transitionDuration = getTransitionDurationFromElement(activeElement)

      EventHandler.one(activeElement, 'transitionend', () => {
        nextElement.classList.remove(directionalClassName, orderClassName)
        nextElement.classList.add(CLASS_NAME_ACTIVE)

        activeElement.classList.remove(CLASS_NAME_ACTIVE, orderClassName, directionalClassName)

        this._isSliding = false

        setTimeout(() => {
          EventHandler.trigger(this._element, EVENT_SLID, {
            relatedTarget: nextElement,
            direction: eventDirectionName,
            from: activeElementIndex,
            to: nextElementIndex
          })
        }, 0)
      })

      emulateTransitionEnd(activeElement, transitionDuration)
    } else {
      activeElement.classList.remove(CLASS_NAME_ACTIVE)
      nextElement.classList.add(CLASS_NAME_ACTIVE)

      this._isSliding = false
      EventHandler.trigger(this._element, EVENT_SLID, {
        relatedTarget: nextElement,
        direction: eventDirectionName,
        from: activeElementIndex,
        to: nextElementIndex
      })
    }

    if (isCycling) {
      this.cycle()
    }
  }

  _directionToOrder(direction) {
    if (![DIRECTION_RIGHT, DIRECTION_LEFT].includes(direction)) {
      return direction
    }

    if (isRTL()) {
      return direction === DIRECTION_RIGHT ? ORDER_PREV : ORDER_NEXT
    }

    return direction === DIRECTION_RIGHT ? ORDER_NEXT : ORDER_PREV
  }

  _orderToDirection(order) {
    if (![ORDER_NEXT, ORDER_PREV].includes(order)) {
      return order
    }

    if (isRTL()) {
      return order === ORDER_NEXT ? DIRECTION_LEFT : DIRECTION_RIGHT
    }

    return order === ORDER_NEXT ? DIRECTION_RIGHT : DIRECTION_LEFT
  }

  // Static

  static carouselInterface(element, config) {
    let data = Data.get(element, DATA_KEY)
    let _config = {
      ...Default,
      ...Manipulator.getDataAttributes(element)
    }

    if (typeof config === 'object') {
      _config = {
        ..._config,
        ...config
      }
    }

    const action = typeof config === 'string' ? config : _config.slide

    if (!data) {
      data = new Carousel(element, _config)
    }

    if (typeof config === 'number') {
      data.to(config)
    } else if (typeof action === 'string') {
      if (typeof data[action] === 'undefined') {
        throw new TypeError(`No method named "${action}"`)
      }

      data[action]()
    } else if (_config.interval && _config.ride) {
      data.pause()
      data.cycle()
    }
  }

  static jQueryInterface(config) {
    return this.each(function () {
      Carousel.carouselInterface(this, config)
    })
  }

  static dataApiClickHandler(event) {
    const target = getElementFromSelector(this)

    if (!target || !target.classList.contains(CLASS_NAME_CAROUSEL)) {
      return
    }

    const config = {
      ...Manipulator.getDataAttributes(target),
      ...Manipulator.getDataAttributes(this)
    }
    const slideIndex = this.getAttribute('data-bs-slide-to')

    if (slideIndex) {
      config.interval = false
    }

    Carousel.carouselInterface(target, config)

    if (slideIndex) {
      Data.get(target, DATA_KEY).to(slideIndex)
    }

    event.preventDefault()
  }
}

/**
 * ------------------------------------------------------------------------
 * Data Api implementation
 * ------------------------------------------------------------------------
 */

EventHandler.on(document, EVENT_CLICK_DATA_API, SELECTOR_DATA_SLIDE, Carousel.dataApiClickHandler)

EventHandler.on(window, EVENT_LOAD_DATA_API, () => {
  const carousels = SelectorEngine.find(SELECTOR_DATA_RIDE)

  for (let i = 0, len = carousels.length; i < len; i++) {
    Carousel.carouselInterface(carousels[i], Data.get(carousels[i], DATA_KEY))
  }
})

/**
 * ------------------------------------------------------------------------
 * jQuery
 * ------------------------------------------------------------------------
 * add .Carousel to jQuery only if jQuery is present
 */

defineJQueryPlugin(NAME, Carousel)

export default Carousel
