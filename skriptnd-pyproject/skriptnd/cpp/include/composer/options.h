/*
 * Copyright (c) 2017-2025 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _SKND_OPTIONS_H_
#define _SKND_OPTIONS_H_


namespace sknd
{

    enum CompilerFlags : unsigned
    {
        EliminateTrivialLoops = 0x01,
        EliminateTrivialLocals = 0x02,
        EliminateTrivialBoundedExprs = 0x04,
        UnrollPackLoops = 0x08,
    };

    inline static const unsigned DefaultCompilerFlags = EliminateTrivialLoops | EliminateTrivialLocals | EliminateTrivialBoundedExprs;

}   // namespace sknd


#endif
