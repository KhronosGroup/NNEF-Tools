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
