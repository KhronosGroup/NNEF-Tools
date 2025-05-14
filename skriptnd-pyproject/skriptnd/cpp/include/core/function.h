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

#ifndef _SKND_FUNCTION_H_
#define _SKND_FUNCTION_H_

#include <type_traits>


namespace sknd
{

    template <typename Signature>
    class function_view;


    template <typename Result, typename... Args>
    class function_view<Result(Args...)> final
    {
    public:
        
        template <typename F, typename = std::enable_if_t<std::is_invocable<F,Args...>::value &&
                                        !std::is_same<std::decay_t<F>, function_view>::value>>
        function_view( F&& fn ) noexcept : _ptr{(void*)std::addressof(fn)}
        {
            _erased_fn = [](void* ptr, Args... args) -> Result
            {
                return (*reinterpret_cast<std::add_pointer_t<F>>(ptr))(std::forward<Args>(args)...);
            };
        }

        Result operator()(Args... args) const
            noexcept(noexcept(_erased_fn(_ptr, std::forward<Args>(args)...)))
        {
            return _erased_fn(_ptr, std::forward<Args>(args)...);
        }
        
    private:
        
        void* _ptr;
        Result (*_erased_fn)(void*, Args...);
    };

}   // namespace sknd

#endif
