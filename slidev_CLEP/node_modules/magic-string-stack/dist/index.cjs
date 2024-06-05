'use strict';

const MagicString = require('magic-string');
const remapping = require('@ampproject/remapping');

function _interopDefaultCompat (e) { return e && typeof e === 'object' && 'default' in e ? e.default : e; }

const MagicString__default = /*#__PURE__*/_interopDefaultCompat(MagicString);
const remapping__default = /*#__PURE__*/_interopDefaultCompat(remapping);

var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => {
  __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
  return value;
};
class MagicStringStack {
  constructor(content, _options) {
    this._options = _options;
    /**
     * The stack of MagicString instances.
     * Lastest instance is pushed to the front of the array.
     */
    __publicField(this, "_stack", []);
    /**
     * Prepresents the current MagicString instance.
     * It should be in the this._stack[0]
     */
    __publicField(this, "_current");
    this._current = new MagicString__default(content, this._options);
    this._stack.unshift(this._current);
    const proxy = new Proxy(new MagicString__default(""), {
      get: (_, p, receiver) => {
        if (Reflect.has(this, p))
          return Reflect.get(this, p, receiver);
        const parent = Reflect.get(this._current, p, receiver);
        if (typeof parent === "function") {
          return (...args) => {
            const result = parent.apply(this._current, args);
            if (result === this._current)
              return proxy;
            return result;
          };
        }
        return parent;
      },
      set: (_, p, value) => {
        return Reflect.set(this, p, value, this);
      }
    });
    return proxy;
  }
  /**
   * Commit current changes and reset the `.original` property and the indices.
   *
   * When running `generateMap`, the sourcemaps will be generated and merged into a single sourcemap.
   */
  commit() {
    const newOne = new MagicString__default(this._current.toString(), this._options);
    this._current = newOne;
    this._stack.unshift(newOne);
    return this;
  }
  /**
   * Rollback to the previous commit.
   */
  rollback() {
    if (this._stack.length <= 1)
      throw new Error("Cannot rollback on the first commit");
    this._stack.shift();
    this._current = this._stack[0];
    return this;
  }
  get original() {
    return this._current.original;
  }
  toString() {
    return this._current.toString();
  }
  clone() {
    const s = new MagicStringStack(this._current.toString(), this._options);
    s._stack = this._stack.map((s2) => s2.clone());
    s._current = s._stack[0];
    return s;
  }
  /**
   * Generates a version 3 sourcemap.
   */
  generateMap(options) {
    return new MagicString.SourceMap(this.generateDecodedMap(options));
  }
  /**
   * Generates a sourcemap object with raw mappings in array form, rather than encoded as a string.
   * Useful if you need to manipulate the sourcemap further, but most of the time you will use `generateMap` instead.
   */
  generateDecodedMap(options) {
    if (this._stack.length === 1)
      return this._current.generateDecodedMap(options);
    const SOURCE_PLACEHOLDER = "__magic_string_placeholder__.js";
    const maps = this._stack.map((s) => s.generateDecodedMap({ source: SOURCE_PLACEHOLDER, ...options }));
    const merged = remapping__default(maps, () => null, { decodedMappings: true });
    merged.sources = merged.sources.map((source) => source === SOURCE_PLACEHOLDER ? "" : source);
    delete merged.version;
    return {
      ...merged
    };
  }
}

module.exports = MagicStringStack;
