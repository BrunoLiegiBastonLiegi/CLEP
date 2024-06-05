import * as esbuild from 'esbuild';
import { Options } from './types.js';
import 'markdown-it';
import '@mdit-vue/plugin-component';
import '@mdit-vue/plugin-frontmatter';
import '@mdit-vue/types';
import '@rollup/pluginutils';

declare const _default: (options: Options) => esbuild.Plugin;

export { _default as default };
