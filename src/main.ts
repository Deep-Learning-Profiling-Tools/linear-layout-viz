import { startDemoApp } from '@tensor-viz/viewer-demo';
import { linearLayoutExtensionFactory } from './extensions/linear-layout/extension.js';

// LL-viz owns the linear-layout extension; tensor-viz supplies the generic shell.
startDemoApp({ extensionFactories: [linearLayoutExtensionFactory] });
